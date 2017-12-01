#!/usr/bin/env gosh

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

;; Map dataset points to nearest clusters.
;; ((('label (x...))...) ((x...)...)) => (('label distance)...)
(define (segment clusters dataset)
  (map (lambda (point)
         (car (sort (map (lambda (c)
                           (list (car c)
                                 (euclidean-distance (cadr c) point)))
                         clusters)
                    (lambda (a b)
                      (<= (cadr a) (cadr b))))))
       dataset))

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

(define (nmeans points)
  (cons (pick-favorite (map car points))
        (map (lambda (x) (/ x (length points)))
             (reduce (lambda (p1 p2)
                       (map + p1 p2))
                     (make-list (length (list-ref points 0)) 0)
                     (map cdr points)))))

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

(define (get-intra-cluster-distance centroid cluster)
  (reduce + 0 (map (lambda (p)
                     (euclidean-distance p centroid))
                   cluster)))

;; => (centroid-list )
;; centroid-list: (label total-inner-distance coordinate)
(define (kmeans-with-k k points bounds)
  (map
   (lambda (attempt-i)
     ;(format (current-error-port) "K~a ~a / ~a.~%" k (+ 1 attempt-i) kmeans-attempts-per-k)
     (flush-all-ports)
     (display #\. (current-error-port))
     (flush-all-ports)
     (letrec* ((reroll (let/cc cc cc))
               (centroids
                (map (lambda (centroid-i)
                       (cons 'IF-YOU-ARE-READING-THIS-YOU-DONT-HAVE-A-VALID-SEASON (nrand bounds)))
                     (iota k))))
       (let loop ((last-centroids centroids))
         (letrec* ((new-centroids-and-clusters
                     (map (lambda (c)
                            (list (nmeans c) c))
                          (or (cluster-it last-centroids points)
                              (reroll reroll))))
                   (new-centroids (map car new-centroids-and-clusters))
                   (clusters (map cadr new-centroids-and-clusters)))
           (if (equal? new-centroids last-centroids)
               `((centroids ,(map (lambda (ct cl)
                                    `((location ,(cdr ct))
                                      (label    ,(car ct))
                                      (size     ,(length cl))))
                                  new-centroids
                                  clusters))
                 (distance  ,(reduce + 0 (map get-intra-cluster-distance
                                              new-centroids
                                              clusters))))
               (loop new-centroids))))))
   (iota kmeans-attempts-per-k)))

(define (best-kmeans-for-k k points bounds)
  (car (sort (kmeans-with-k k points bounds)
             (lambda (a b)
               (<= (cadr (assoc 'distance a))
                   (cadr (assoc 'distance b)))))))

(define (kmeans k points)
  (letrec ((bounds (nbounds points))
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
          (format #t ;#(current-error-port)
                  "K\tBest ICD~%")
          (display (format-result k (best-kmeans-for-k k points bounds) #t)))

        ;; else print ICD for each K
        (begin
          (format #t ;#(current-error-port)
                  "K\tBest ICD~%")
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
