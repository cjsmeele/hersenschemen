#!/usr/bin/env gosh

;; kmeans.scm - K-means clustering data classifier
;; Copyright (C) 2017, Chris Smeele and Jan Halsema.
;;
;; This program is free software: you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation, either version 3 of the License, or
;; (at your option) any later version.
;;
;; This program is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;; GNU General Public License for more details.
;;
;; You should have received a copy of the GNU General Public License
;; along with this program. If not, see <https://www.gnu.org/licenses/>.

(use srfi-27) ;; For random numbers.

;; Constants.
(define kmeans-attempts-per-k  10)
(define kmeans-max-auto-k      10)

;; Scheme port of Perl's yada yada operator.
(define (...)
  (error "Het staat op Canvas, zoek het uit.\n"))

;; We don't want line-buffering to mess up our fancy
;; '......' progress output.
(set! (port-buffering (current-error-port))  :none)
(set! (port-buffering (current-output-port)) :none)

;; CSV filename => ((cell...)...)
;; Cells are converted to numbers.
(define (parse-csv-file filename)
  (let loop ((p (open-input-file filename))
             (data '()))
    (let ((v (read-line p)))
      (if (eof-object? v)
          (reverse data)
          (loop p (cons (map string->number (string-split v #\;))
                        data))))))

;; Calculate the distance between two points in n-dimensional space.
(define (euclidean-distance p1 p2)
  (sqrt (reduce + 0 (map (lambda (x) (expt x 2))
                         (map - (cdr p1) (cdr p2))))))

;; Map a YYYYMMDD date number (e.g. 20170131) to a season symbol.
(define (date-to-season date-nr)
  ;; Extract the month and match it.
  (ecase (div (modulo date-nr 10000) 100)
    ((12  1  2) 'winter)
    (( 3  4  5) 'spring)
    (( 6  7  8) 'summer)
    (( 9 10 11) 'autumn)))

;; Determine the bounds for each dimension in dataset.
;; ((x...)...) => ((min max)...)
(define (nbounds dataset)
  (let ((cardinality (length (cdar dataset))))
    (map (lambda (i)
           (let ((values (map (lambda (row) (list-ref row (+ 1 i)))
                              dataset)))
             (list (apply min values)
                   (apply max values))))
         (iota cardinality))))

;; Generate a random point within bounds.
;; ((min max)...) => (rand...)
(define (nrand bounds)
  (map (lambda (b)
         (+ (car b)
            (random-integer (+ 1 (- (cadr b) (car b))))))
       bounds))

;; From a list, return the value that occurs the most often.
;; Keep popping the list until exactly one value has the most occurrences.
(define (pick-favorite ordered-values)
  (letrec* ((uniques (delete-duplicates ordered-values))
            (uniques-counts
             (map (lambda (uv) (count (lambda (ov) (equal? ov uv)) ordered-values))
                  uniques))
            (count-max (apply max uniques-counts)))
    (if (= 1 (count (lambda (c) (= c count-max)) uniques-counts))
        (any (lambda (v c) (and (= count-max c) v))
             uniques
             uniques-counts)
        (pick-favorite (take ordered-values
                             (- (length ordered-values) 1))))))

;; Determine the center of a list of n-dimensional points.
;; ((x...)...) => (center-x...)
(define (nmeans points)
  (cons (pick-favorite (map car points))
        (map (lambda (x) (/ x (length points)))
             (reduce (lambda (p1 p2)
                       (map + p1 p2))
                     (make-list (length (list-ref points 0)) 0)
                     (map cdr points)))))

;; Given a list of centroids and a list of points,
;; generate a list of points for each centroid.
;; The order of points within each cluster list is undefined.
;; (((ctr-x...)...) ((x...)...) => (((x...)...)...)
(define (cluster-it centroids points)
  (let ((clusters (make-list (length centroids) '())))
    (for-each
     (lambda (p)
       (let ((ci (caar (sort (map (lambda (i cp)
                                    (list i (euclidean-distance (cdr cp) (cdr p))))
                                  (iota (length centroids))
                                  centroids)
                             (lambda (a b)
                               (<= (cadr a) (cadr b)))))))
         (list-set! clusters ci
                    (cons p (list-ref clusters ci)))))
     points)
    (if (= 0 (count (lambda (c) (= 0 (length c))) clusters))
        clusters
        #f)))

;; Determine the total intra-cluster distance.
;; ((ctr-x...) ((x...)...)) => distance
(define (get-intra-cluster-distance centroid cluster)
  (reduce + 0 (map (lambda (p)
                     (euclidean-distance p centroid))
                   cluster)))

;; Generate an unsorted list of length `kmeans-attempts-per-k',
;; containing centroid info and a distance number for each attempt.
;;
;; (k ((x...)...) ((min max)...))
;; => (('centroids (('location (x...))
;;                  ('label season)
;;                  ('size nr-of-points-in-cluster))
;;     ('distance total-icd))...)
(define (kmeans-with-k k points bounds)
  (map ;; Map over 1..kmeans-attempts-per-k
   (lambda (attempt-i)
     ;(format (current-error-port) "K~a ~a / ~a.~%" k (+ 1 attempt-i) kmeans-attempts-per-k)
     (flush-all-ports)
     (display #\. (current-error-port))
     (flush-all-ports)

     (letrec* (;; Calling (reroll reroll) within the let body jumps back to this point.
               ;; This way we can create new initial centroids if we find an empty cluster.
               (reroll (let/cc cc cc))
               ;; Create initial centroids.
               (centroids
                (map (lambda (centroid-i)
                       (cons 'IF-YOU-ARE-READING-THIS-YOU-DONT-HAVE-A-VALID-SEASON (nrand bounds)))
                     (iota k))))

       ;; Keep reclustering and recentering until the centroids no longer move.
       (let loop ((last-centroids centroids))
         (letrec* ((new-centroids-and-clusters
                     (map (lambda (c)
                            (list (nmeans c) c))
                          (or (cluster-it last-centroids points)
                              (reroll reroll))))
                   (new-centroids (map car new-centroids-and-clusters))
                   (clusters (map cadr new-centroids-and-clusters)))

           ;; Assert that the centroid position is stored as a list of rational numbers.
           ;; This way we can safely compare centroid positions for equality without
           ;; messy floating point precision issues.
           (when (any (lambda (new old) (or (inexact? new)
                                            (inexact? old)))
                      new-centroids last-centroids)
             (error "Centroid positions are not exact!"))

           (if (equal? new-centroids last-centroids)
               ;; Stable! Return the centroids and the total ICD.
               `((centroids ,(map (lambda (ct cl)
                                    `((location ,(cdr ct))
                                      (label    ,(car ct))
                                      (size     ,(length cl))))
                                  new-centroids
                                  clusters))
                 (distance  ,(reduce + 0 (map get-intra-cluster-distance
                                              new-centroids
                                              clusters))))
               ;; Try again.
               (loop new-centroids))))))
   (iota kmeans-attempts-per-k)))

;; Call kmeans-with-k and return the result with the lowest ICD.
(define (best-kmeans-for-k k points bounds)
  (car (sort (kmeans-with-k k points bounds)
             (lambda (a b)
               (<= (cadr (assoc 'distance a))
                   (cadr (assoc 'distance b)))))))

;; Generate the given amount of clusters with the given points.
;; If K is a number, generate that many clusters and print the centroids and their seasons.
;; If K is #f, generate K clusters for K in 1..kmeans-max-auto-k and print their best ICDs.
(define (kmeans k points)
  (let ((bounds (nbounds points))
        (format-result
         (lambda (k r dump-centroids?)
           (let ((ct (cadr (assoc 'centroids r))))
             (format "~a\t~a~%~a"
                     k
                     (inexact (cadr (assoc 'distance r)))
                     (if dump-centroids?
                         (reduce string-append ""
                                 (map (lambda (ct)
                                        (format "~a(~a) @ ~a~%"
                                                (cadr (assoc 'label ct))
                                                (cadr (assoc 'size ct))
                                                (map inexact (cadr (assoc 'location ct)))))
                                      ct))
                         ""))))))
    (if k
        (begin
          (display "K\tBest ICD\n")
          (display (format-result k (best-kmeans-for-k k points bounds) #t)))
        (begin
          (display "K\tBest ICD\n")
          (for-each
           (lambda (k)
             (display (format-result k (best-kmeans-for-k k points bounds) #f)))
           (iota (min kmeans-max-auto-k (length points)) 1))))))


(define (usage program-name)
  (format (current-error-port)
          "usage: ~a <auto | K> DATASET-CSV\n"
          program-name)
  (exit 2))

(define (main args)
  (random-source-randomize! default-random-source)
  (when (not (= (length args) 3))
    (usage (car args)))
  (kmeans (if (string=? "auto" (cadr args))
              #f
              (string->number (cadr args)))
          (map (lambda (p)
                 (cons (date-to-season (car p))
                       (cdr p)))
               (parse-csv-file (caddr args)))))
