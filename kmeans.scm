#!/usr/bin/env gosh

(use srfi-27) ;; For random numbers.

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

(define (kmeans k dataset)
  (let ((cardinality (length (car dataset)))
        (bounds (nbounds dataset)))
    (if k
        (display "UNIMPLEMENTED\n")
        ;; TODO.
        #t)))

(define (usage program-name)
  (format (current-error-port)
          "usage: ~a <auto | K> DATASET-CSV\n"
          program-name)
  (exit 2))

(define (main args)
  (when (not (= (length args) 3))
    (usage (car args)))
  (kmeans (if (string=? "auto" (cadr args))
              #f
              (string->number (cadr args)))
          (parse-csv-file (caddr args))))
