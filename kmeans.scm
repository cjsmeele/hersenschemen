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
                         (map - p1 p2)))))

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
;; XXX: UNTESTED
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
  (let ((cardinality (length (car dataset))))
    (map (lambda (i)
           (let ((values (map (lambda (row) (list-ref row i))
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

(define (nmeans points)
 (map (lambda (x) (/ x (length points)))
       (reduce (lambda (p1 p2)
                 (map + p1 p2))
               (make-list (length (list-ref points 0)) 0)
               points)))

(define (cluster-it centroids points)
  (let ((clusters (make-list (length centroids) '())))
    ;;(display centroids)
    (for-each
     (lambda (p)
       ;; (display "p")
       (let ((ci (caar (sort (map (lambda (i cp)
                               ;; (format #t "~a ~a ~a~%" i cp (euclidean-distance cp p))
                                    (list i (euclidean-distance cp p)))
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

(define kmeans-attempts-per-k 1)

(define (kmeans k points)
  (let ((cardinality (length (car points)))
        (bounds (nbounds points)))
    (let ((kmeans-with-k
           (lambda (k)
             (let ((clusters
                    (map
                     (lambda (attempt-i)
                       (letrec* ((reroll (let/cc cc cc))
                                 (centroids
                                  (map (lambda (centroid-i)
                                         (nrand bounds))
                                       (iota k))))
                         (display "Initially (re)rolled centroids:\n")
                         (for-each
                          (lambda (c)
                            (display c)
                            (newline))
                          centroids)
                         (let loop ((last-centroids centroids))
                           (display "Reclustering / Recentering\n")
                           (let ((new-centroids
                                  (map nmeans
                                       (or (cluster-it last-centroids points)
                                           (reroll reroll)))))
                             (for-each
                              (lambda (c)
                                (display (map inexact c))
                                (newline))
                              new-centroids)
                             (newline)
                             (if (equal? new-centroids last-centroids)
                                 ;; (list centroids clusters)
                                 (begin (display "STABLE! :D\n\n") centroids)
                                 (loop new-centroids))))))
                     (iota kmeans-attempts-per-k))))))))
      (if k
          (kmeans-with-k k)
          ;; else find optimal K
          ;; TODOOOO
          (...)
          ))))


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
          (map cdr (parse-csv-file (caddr args)))))
