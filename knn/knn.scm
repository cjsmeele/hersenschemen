#!/usr/bin/env gosh

;; CSV filename => 2d list by row,col.
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

;; Map a YYYYMMDD date number (e.g. 20170131) to a season symbol.
(define (date-to-season date-nr)
  ;; Extract the month and match it.
  (ecase (div (modulo date-nr 10000)
             100)
    ((12  1  2) 'winter)
    (( 3  4  5) 'spring)
    (( 6  7  8) 'summer)
    (( 9 10 11) 'autumn)))

;; Given base data with known dates, and input data with unknown dates,
;; determine the season for each input datum using the KNN algorithm.
(define (classify-it k base-data input-data)
  (let ((base-seasons (map (lambda (entry)
                             (date-to-season (car entry)))
                           base-data)))
    (map (lambda (input)
           (pick-favorite
            (take ;; Take K seasons from list ordered by distance.
             (map cadr
                  (sort
                   ;; => ((distance season)...)
                   (map (lambda (base its-season)
                          (list (euclidean-distance (cdr base) (cdr input))
                                its-season))
                        base-data
                        base-seasons)
                   (lambda (a b)
                     ;; Sort by distance, near to far.
                     (<= (car a) (car b)))))
             k)))
         input-data)))

;; Run classify-it with data read from CSV files and print the results.
(define (classify k base-csv input-csv)
  (let ((input-seasons
         (classify-it k
                      (parse-csv-file base-csv)
                      (parse-csv-file input-csv))))
    (for-each
     (lambda (l)
       (display l)
       (newline))
     input-seasons)))

;; Run classify with validation data as the input, of which the dates are known.
;; Return a correctness ratio.
(define (verify-it k base-data validation-data)
  (/ (count equal?
            (classify-it k base-data validation-data)
            (map date-to-season (map car validation-data)))
     (length validation-data)))

(define (verify k base-csv validation-csv)
  (format #t "~a%~%" (floor (* 100
                               (verify-it k
                                          (parse-csv-file base-csv)
                                          (parse-csv-file validation-csv))))))

;; For 1..max-k, determine for each K the correctness value returned by verify.
;; => ((K correctness)...)
(define (grade-k-it max-k base-data validation-data)
  (map (lambda (n)
         (list n (verify-it n base-data validation-data)))
       (iota (apply min (list max-k (length base-data))) 1)))

;; Run grade-k-it with K values 1..100 and print a list of tab-separated
;; k / correctness values.
(define (grade-k base-csv validation-csv)
  (for-each
   (lambda (kg)
     (format #t "~a\t~a~%"
             (car kg)
             (inexact (cadr kg))))
   (grade-k-it
    100
    (parse-csv-file base-csv)
    (parse-csv-file validation-csv))))

(define (usage program-name)
  (format (current-error-port)
          "usage: ~a <classify K | verify K | grade-k> BASE-CSV <INPUT-CSV | VALIDATION-CSV>\n"
          program-name)
  (exit 2))

(define (main args)
  (when (< (length args) 2)
    (usage (car args)))
  (cond ((and (string=? "classify" (cadr args))
              (= 5 (length args)))
         (apply classify (cons (string->number (caddr args))
                               (cdddr args))))
        ((and (string=? "verify" (cadr args))
              (= 5 (length args)))
         (apply verify (cons (string->number (caddr args))
                             (cdddr args))))
        ((and (string=? "grade-k" (cadr args))
              (= 4 (length args)))
         (apply grade-k (cddr args)))
        (else (usage (car args)))))
