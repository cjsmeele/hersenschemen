#!/usr/bin/env gosh

;; wings.scm - Genetic algorithm implementation
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

;; Parameters.
(define ga-pop             8) ;; Total population size.
(define ga-plebs         4/8) ;; These will be purged.
(define ga-elite         4/8) ;; These will pair up (randomly) and produce offspring for the next cycle.
(define ga-elite-elders  2/4) ;; The best of the elite will survive an evolution cycle as-is.
                              ;; (elders ⊆ elite)

(define (assert-sane-parameters)
  (format #t "ga-pop:          ~a~%"      ga-pop)
  (format #t "ga-plebs:        ~a (~a)~%" ga-plebs (* ga-plebs ga-pop))
  (format #t "ga-elite:        ~a (~a)~%" ga-elite (* ga-elite ga-pop))
  (format #t "ga-elite-elders: ~a (~a)~%" ga-elite-elders (* ga-elite-elders ga-elite ga-pop))
  (format #t "offspring:       ~a~%"      (- ga-pop (* ga-elite-elders ga-elite ga-pop)))

  (when (not (equal? (+ ga-plebs ga-elite) 1))
    (error "GA constants plebs+elite do not add up"))

  (when (not (every integer?
                    (list (* ga-plebs ga-pop)
                          (* ga-elite ga-pop)
                          (* ga-elite-elders ga-elite ga-pop))))
    (error "GA pop fractions do not produce integer number of individuals"))

  (when (not (even? (- (* ga-elite ga-pop) (* ga-elite-elders ga-elite ga-pop))))
    (error "GA (elite - elders) count must be an even number"))

  (when (not (even? (- ga-pop (* ga-elite-elders ga-elite ga-pop))))
    (error "GA offspring count must be an even number")))

;; Generate random number within bounds.
(define (rand min max)
  (random-integer (+ 1 (- max min))))

;; Generate 2 unique random numbers within bounds.
(define (rand2 min max)
  (let ((a (rand min max))
        (b (rand min max)))
    (if (not (= a b))
        (list a b)
        (rand2 min max))))

;; Generate random boolean.
(define (randb)
  (> (random-integer 2) 0))

;; Generate a random point within bounds.
;; ((min max)...) => (rand...)
(define (nrand bounds)
  (map (lambda (b)
         (+ (car b) (rand (car b) (cadr b))))
       bounds))

;; Lift = (A − B)^2 + (C + D)^2 − (A − 30)^3 − (C − 40)^3
(define (get-lift A B C D)
  (- (+ (expt (- A B) 2)
        (expt (+ C D) 2))
     (expt (- A 30) 3)
     (expt (- C 40) 3)))

(define (get-fitness I)
  ;; Do you even...?
  (apply get-lift I))

(define (mutate! I)
  (let ((mutate-chromosome
         (lambda (c)
           ;; ABCD als bitstrings benaderen ipv als integers is IMHO onlogisch,
           ;; maar hé, opdracht is opdracht.
           ;; Flip a random bit.
           (logxor c (ash 1 (rand 0 5))))) ;; Keep C between 0-63, incl.
        (count (rand 1 4)))
    (dotimes (count)
      (let ((r (rand 0 3)))
        (set! (list-ref I r) (mutate-chromosome (list-ref I r)))))))

;; Produce two offspring each with the other half of the randomly mixed
;; chromosomes of their parents.
(define (vertical-gene-transfer I1 I2)
  (let ((division (map (lambda (_) (randb)) I1)))
    (if (> (length (delete-duplicates division)) 1)
        (list (map (lambda (d C1 C2) (if d C1 C2)) division I1 I2)
              (map (lambda (d C1 C2) (if d C2 C1)) division I1 I2))
        ;; Make sure we're not producing 2 exact clones, try again.
        (vertical-gene-transfer I1 I2))))

;; Run a single generation.
(define (ga-generation Is gen-count)
  ;; Sort current generation by fitness.
  (let ((Is-o (sort Is (lambda (a b)
                         (> (get-fitness a)
                            (get-fitness b))))))
    (if (> gen-count 0)
        (begin
          (format #t "~%:: Generation A-~a~%" gen-count)
          ;; Cull the weak.
          (let* ((elite  (take       Is-o  (* ga-elite ga-pop)))
                 (elders (take       elite (* ga-elite-elders ga-elite ga-pop)))
                 (plebs  (take-right Is-o  (* ga-plebs ga-pop))))

            (format #t "- Elite~%")
            (dolist (I elite) (format #t "~a ~a~%" I (get-fitness I)))
            (format #t "- Plebs~%")
            (dolist (I plebs) (format #t "~a ~a~%" I (get-fitness I)))

            (let loop ((ng elders))
              (if (< (length ng) ga-pop)
                  ;; Produce offspring.
                  (loop (append
                         (apply vertical-gene-transfer
                                (map (lambda (n) (list-ref elite n))
                                     (rand2 0 (- (length elite) 1))))
                         ng))
                  (begin
                    ;; Mutate offspring + elders.
                    (map mutate! ng)
                    (ga-generation ng (- gen-count 1)))))))
          (car Is-o))))

(define (ga gcount)
  (let ((winner
         (ga-generation
          (map (lambda (n)
                 (nrand '((0 63) (0 63) (0 63) (0 63))))
               (iota ga-pop))
          gcount)))
    (format #t "~%And the winner, with fitness ~a, is:~%~a~%"
            (get-fitness winner) winner)))

(define (usage program-name)
  (format (current-error-port)
          "usage: ~a GENERATION-SIZE GENERATION-COUNT\n"
          program-name)
  (exit 2))

(define (main args)
  (random-source-randomize! default-random-source)
  (when (not (= (length args) 3))
    (usage (car args)))
  (set! ga-pop (string->number (cadr args)))
  (assert-sane-parameters)
  (ga (string->number (caddr args))))
