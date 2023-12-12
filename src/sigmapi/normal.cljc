(ns sigmapi.normal
  (:require
    [clojure.stacktrace :refer [print-cause-trace]]
    [clojure.math :as maths :refer [PI]]
    [sigmapi.core :as sp :refer :all :rename {log2 logg2}]
    [clojure.core.matrix :as m]
    [emmy.env :as e :refer :all]
    [emmy.matrix :as em]))


(defn normal-log [mu sd]
  (fn [x]
     (/
       (+
        (* -1 (expt sd 2) (log sd))
        (* -1/2 (expt mu 2))
        (* mu x)
        (* -0.9189385332046727 (expt sd 2))
        (* -1/2 (expt x 2)))
       (expt sd 2))))

(defn multivariate-normal-log [mu sigma]
  (fn [x]
    (let [sigma (apply em/by-rows sigma)
          mu (apply em/column mu)
          k (count x)
          x (apply em/column x)
          x-mu (- x mu)
          sigma-1 (em/invert sigma)
          det-sigma (em/determinant sigma)]
      (em/get-in
        (+
          (* -1/2 (em/transpose x-mu) sigma-1 x-mu)
          (* -0.9189385332046727 k)
          (* -1/2 (log det-sigma)))
        [0 0]))))

(defn multivariate-normal [mu sigma]
  (fn [x]
    (let [sigma (apply em/by-rows sigma)
          mu (apply em/column mu)
          k (count x)
          x (apply em/column x)
          x-mu (- x mu)
          sigma-1 (em/invert sigma)
          det-sigma (em/determinant sigma)]
      (em/get-in
        (/
          (exp (* -1/2 (em/transpose x-mu) sigma-1 x-mu))
          (sqrt (* (expt (* 2 PI) k) det-sigma)))
        [0 0]))))

(deftype NormalVariableNode [id]
  sp/Messaging
  (>< [this messages to]
    (println " v><" id (map (juxt :id :value) messages))
    {
     :value     (map * (map :value messages))
     :repr      (if (== 1 (count messages)) (:repr (first messages)) (cons '∏ (map :repr messages)))
     })
  (<> [this messages to to-msg parent-msg]
    (print "  v<>" id)
    (>< this messages to))
  (i [this]
    {:value 0 :repr id})
  sp/Variable
  sp/Passes
  (pass? [this] false))

(deftype NormalFactorNode
  [f id dim-for-node]
  sp/Messaging
  (>< [this messages to]
    (println " f><" id (map (juxt :id :value) messages))
    (let [
          ; pointwise product of functions
          prod (map * (map :value messages))
          sum  (map + prod)
          ]
      {
       :value     sum
       :repr      (cons '∑ (list (cons '∏ (list (:repr (i this)) (if (== 1 (count messages)) (:repr (first messages)) (map :repr messages))))))
       }))
  (<> [this messages to to-msg parent-msg]
    (print "  f<>" id)
    (>< this messages to))
  (i [this]
    {:value f :repr id :dim-for-node dim-for-node})
  sp/Factor
  sp/Passes
  (pass? [this] true))

(defmethod make-node [:sp/sp :sp/normal-variable]
  ([{id :id}]
    (NormalVariableNode. id)))

(defmethod make-node [:sp/sp :sp/normal-factor]
  ([{:keys [graph id clm cpm dfn]}]
    (NormalFactorNode. cpm id dfn)))




(comment




(simplify
  (log
    (*
      (/ 1 (* 'sd (sqrt (* 2 PI))))
      (exp (* -1/2 (expt (/ (- 'x 'mu) 'sd) 2))))))


'(/
   (+
    (* -1 (expt sd 2) (log sd))
    (* -1/2 (expt mu 2))
    (* mu x)
    (* -0.9189385332046727 (expt sd 2))
    (* -1/2 (expt x 2)))
   (expt sd 2))


  (let [f (fn [mu sd]
            (fn [x]
               (/
                 (+
                  (* -1 (expt sd 2) (log sd))
                  (* -1/2 (expt mu 2))
                  (* mu x)
                  (* -0.9189385332046727 (expt sd 2))
                  (* -1/2 (expt x 2)))
                 (expt sd 2))))
        f+ (apply + (map :v [{:v (f 0 1)} {:v (f 0 1)}]))
        ]
    (f+ 0.5))



  (m/mmul [0.3 0.7] [[1 0.5] [0.5 1]] [0.9 0.4])

  (simplify
    (log
      (/
        (exp (* -1/2 'x-muT 'sigma-1 'x-mu))
        (sqrt (* (expt (* 2 PI) 'k) 'det-sigma)))))

(map (comp + log) (range 7))

  ((multivariate-normal-log [0 0] [[1 0.5] [0.5 1]]) [0 0])


  (* (expt (apply em/column [1 0.3]) 2) (apply em/by-rows [[1 0.5] [0.5 1]]))

(print-cause-trace *e)

(let
    [model
     {:fg
      (sp/fgtree
        (:d [:pd [0.5 0.5]]
          [:h|d
           [[0.6 0.5 0.4] [0.8 0.5 0.2]]
           (:h)
           ]))
      :priors
      {:d :pd}}
     ]
  ;(i (get-in (exp->fg :sp/sp (:fg model)) [:nodes :d]))
  (->>
    (reductions
      (fn [{{p :dp} :marginals :as m} {d :pd :as data}]
          (update-priors (assoc m :data data)))
        model
       [{:pd [0 1]}])
      (map :marginals)
      ))






)