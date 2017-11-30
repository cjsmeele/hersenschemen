#!/usr/bin/env gosh

(use srfi-27) ;; For random numbers.

(define (...)
  (error "Het staat op Canvas, zoek het uit.\n"))

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
  (ecase (div (modulo date-nr 10000)
             100)
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
  ;(display (car dataset))
  ;(newline)
  ;(display (cdar dataset))
  ;(newline)
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
    ;;(display centroids)
    (for-each
     (lambda (p)
       ;; (display "p")
       (let ((ci (caar (sort (map (lambda (i cp)
                               ;; (format #t "~a ~a ~a~%" i cp (euclidean-distance cp p))
                                    (list i (euclidean-distance (cdr cp) (cdr p))))
                                  (iota (length centroids))
                                  centroids)
                             (lambda (a b)
                               (<= (cadr a) (cadr b)))))))
         ;; (display ci)
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

(define kmeans-attempts-per-k 100)

;; => (centroid-list )
;; centroid-list: (label total-inner-distance coordinate)
(define (kmeans-with-k k points bounds)
  (map
   (lambda (attempt-i)
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
               (begin (format #t "Calculated ~a / ~a.~%" (+ 1 attempt-i) kmeans-attempts-per-k)
                      ; '((key val) (key val))
                      `((centroids ,(map (lambda (ct cl)
                                           `((location ,(cdr ct))
                                             (label    ,(car ct))
                                             (size     ,(length cl))))
                                         new-centroids
                                         clusters))
                        (distance  ,(reduce + 0 (map get-intra-cluster-distance
                                                     new-centroids
                                                     clusters)))))
                      ;;(list new-centroids
                      ;;      (reduce + 0 (map get-intra-cluster-distance
                      ;;                       new-centroids
                      ;;                       clusters))
                      ;;      (map length clusters)))
               (loop new-centroids))))))
   (iota kmeans-attempts-per-k)))

(define (kmeans k points)
  (let ((bounds (nbounds points)))
    (if k
      (let ((results (kmeans-with-k k points bounds)))
        (for-each
          (lambda (r)
            (let ((ct (cadr (assoc 'centroids r))))
              (format #t "~a ~a~%"
                      (map cdr ct)
                      (inexact (cadr (assoc 'distance r))))))
          (sort results
                (lambda (a b)
                  (<= (cadr (assoc 'distance a))
                      (cadr (assoc 'distance b)))))))
      ;; else find optimal K
      ;; TODOOOO
      (...))))


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
          ;; (map cdr (parse-csv-file (caddr args)))))
          (map (lambda (p)
                 (cons (date-to-season (car p))
                       (cdr p)))
               (parse-csv-file (caddr args)))))
