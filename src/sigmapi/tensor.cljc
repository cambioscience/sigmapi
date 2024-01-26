(ns sigmapi.tensor
  (:require
    [clojure.core.matrix :as m]
    [clojure.set :as set]
    [clojure.math :as maths :refer [log pow]]
    [sigmapi.core :refer :all]))



(def log2 (log 2))

(defn ln [x] (/ (log x) log2))

(def ln- (comp (partial * -1) ln))

(defn P [x] (pow 2 (* -1 x)))

(defn normalize
  ([p]
    (normalize p (reduce + p)))
  ([p s]
   (if (zero? s)
    p
     (mapv (partial * (/ 1 s)) p))))

(defn random-matrix
  "Returns a random matrix of the given shape e.g.  [2 3 4 5]"
  [[f & r]]
  (if (nil? r)
    (repeatedly f rand)
    (repeatedly f (partial random-matrix r))))

(defn indexed-best
  "
    Returns list of the best (according to the given function f)
    items in the given matrix, and their indices.
  "
  [f]
  (fn ibf [mat]
    (let [best (f mat)]
      [best (first (filter (fn [v] (== best (apply m/mget mat v))) (m/index-seq mat)))])))

(def indexed-min (indexed-best m/emin))

(def indexed-max (indexed-best m/emax))

(defn rotate-vec
  "
    Rotate the given vector v so that index
    i is at the position given by fn f (first or last)
  "
  [v i f]
  (if (= f last)
    (into (subvec v (inc i) (count v)) (subvec v 0 (inc i)))
    (into (subvec v i (count v)) (subvec v 0 i))))

(defn tranz
  "
    Return the given matrix mat transposed such that the dimension at index i
    is the first or last (f) dimension.
    e.g. if m is a matrix with shape [2 3 4 5], then (tranz m 2 last) has shape [5 2 3 4]
          or (tranz m 2 first) has shape [4 5 2 3]
    Don't like this, there must be a better way.
  "
  ([mat i f]
    (tranz mat (vec (range (m/dimensionality mat))) i f))
  ([mat v i f]
   (let [rv (rotate-vec v i f)
         tm (m/transpose mat rv)
         ]
    [tm rv
      (rotate-vec v
        (mod (- (dec (m/dimensionality mat)) i) (m/dimensionality mat))
        (if (= last f) first last))])))

; mat is a matrix, g is the operation for combining matrices
(defn combine
  "
  Returns the product of the given messages
  using the given function g (e.g. add).
  Each message's value will have a different dimension
  so the matrix is transposed so that its last dimension
  matches. (Broadcast the message's vector would still involve
  transposing)
  Finally the result is transposed so the dimension of
  the destination node to is the first dimension, ready
  for summing. Hmm maybe that last bit should be a separate fn
  "
  [mat g messages to dim-for-node]
  (let
    [
     dimz (vec (range (m/dimensionality mat)))
     [p pv ddd]
       (reduce
         (fn [[r pv pnd] {id :id v :value}]
           (let [
                 d (get pnd (dim-for-node id))
                 [tm rv nd] (tranz r dimz d last)
                 tm #?(:clj (if (vector? tm) (m/matrix tm) tm)
                       :cljs tm)
                 q (g tm v)]
             [q rv (mapv pnd nd)]))
         [mat dimz dimz] messages)
       d (get ddd (dim-for-node to))
       [tm rv nd] (tranz p dimz d first)
     ]
    tm))


(defmethod make-node [:MAP :factor :tensor]
  ([{:keys [clm cpm] :as node}]
   (let [node (-> node
                (assoc :f (or clm (m/emap ln- cpm)) :kind :factor)
                (update :features conj :passes))]
     (with-meta node
       {
        ; Messaging
        `><
        (fn [{:keys [f id dim-for-node] :as this} messages to]
          (let [
                prod (combine f m/add messages to dim-for-node)
                mm (map m/emin prod)
                ]
            {
             :dim-for-node dim-for-node
             :value mm
             :min (indexed-min mm)
             :sum prod
             :im (mapv (fn [[s c]] [s (zipmap (keys (dissoc dim-for-node to)) c)]) (map indexed-min prod))
             :repr (list 'min (cons '∑ (cons (:repr (i this)) (map :repr messages))))
             }))
        `<>
        (fn [{:keys [f id dim-for-node] :as this} messages to to-msg parent-msg]
          (let [
                conf (get-in parent-msg [:configuration id])
                mind (zipmap (map :id messages) (range (count messages)))
                to-conf (get conf to)
                ]
            {
             :dim-for-node dim-for-node
             :value 0
             :mind mind
             :conf conf
             :configuration (assoc (:configuration parent-msg) to to-conf)
             }))
        `i (fn [{:keys [f id dim-for-node] :as this}] {:value f :repr id :dim-for-node dim-for-node})
        `updated
          (fn [this {:keys [cpm clm]}]
            (assoc this :f (or clm (m/emap ln- cpm))))
        ; LogSpace
        `p (fn [this x] (m/emap P x))}))))

(defmethod make-node [:MAP :variable :tensor]
  ([{:keys [id] :as node}]
  (with-meta (assoc node :kind :variable)
    {; Messaging
      `><
      (fn [this messages to]
        (let [sum (apply m/add (map :value messages))]
          {
           :value sum
           :repr (cons '∑ (map :repr messages))
           }))
      `<>
      (fn [this messages to to-msg parent-msg]
        (let
          [
           ; to-msg is the msg received by this node from to on the >< pass,
           ; which contains the indices of the other variables for each of this variable's states.
           ; Here we are telling to its configuration and the configurations of all previous variables
           ; In the outflowing messaging, the root variable node uses all its messages
           sum (apply m/add (map :value (cons to-msg messages)))
           min (indexed-min sum)
           ; look up the configuration we got in the forward pass which lead to this minimum
           ; (for the root - others need to use the indices they got from the parent)
           conf (if parent-msg (get-in parent-msg [:configuration id]) (get-in min [1 0]))
           configuration (if parent-msg (:configuration parent-msg) {id conf})
           mto (get-in to-msg [:im conf 1])
           ]
          {
           :value sum
           :min min
           :configuration (assoc configuration to mto)
           :repr (cons '∑ (map :repr messages))
           }))
      `i (fn [this] {:value 0 :repr 0})
      ; LogSpace
      `p (fn [this x] (m/emap P x))})))

(defmethod make-node [:sp :factor :tensor]
  ([{:keys [clm cpm] :as node}]
   (with-meta (-> node
                (assoc :f (or clm (m/emap ln- cpm)) :kind :factor)
                (update :features conj :passes))
     {; Messaging
       `><
       (fn [{:keys [f id dim-for-node] :as this} messages to]
         (let [
               prod (combine f m/add messages to dim-for-node)
               sum (m/emap ln- (map m/esum (m/emap P prod)))
               ]
           {
            :value sum
            :repr (cons '∑ (list (cons '∏ (list (:repr (i this)) (if (== 1 (count messages)) (:repr (first messages)) (map :repr messages))))))
            }))
       `<>
       (fn [this messages to to-msg parent-msg]
         (>< this messages to))
       `i
       (fn [{:keys [f id dim-for-node]}]
         {:value f :repr id :dim-for-node dim-for-node})
      `updated
        (fn [this {:keys [cpm clm] :as p}]
          (assoc this :f (or clm (m/emap ln- cpm))))
       ; LogSpace
       `p (fn [this x] (m/emap P x))})))

(defmethod make-node [:sp :variable :tensor]
  ([node]
   (with-meta (assoc node :kind :variable)
     {; Messaging
       `><
        (fn [this messages to]
           {
            :value (apply m/add (map :value messages))
            :repr (if (== 1 (count messages)) (:repr (first messages)) (cons '∏ (map :repr messages)))
            })
       `<>
        (fn [this messages to to-msg parent-msg]
           (>< this messages to))
       `i (fn [{id :id}] {:value 0 :repr id})
       ; LogSpace
       `p (fn [this x] (m/emap P x))})))

(defn normalize-vals [m]
  (into {}
    (map
      (juxt key
        (comp (fn [v] {:cpm (if (== 1 (m/dimensionality v)) (normalize v) (mapv normalize v))}) val)) m)))

(def marginals
  (comp normalize-vals unnormalized-marginals))

(defn compute-marginals [exp]
  (normalize-vals
    (unnormalized-marginals (propagate (exp->fg :sp exp)))))

(defn update-variables [{nodes :nodes :as graph} post priors data]
  (reductions
    (fn [[g post] data-priors]
      (let [
              p2 (select-keys post (keys priors))
              p1 (merge (zipmap (vals priors) (map p2 (keys priors))) data-priors)
              g  (update-factors g p1)
            ]
        [g (normalize-vals (unnormalized-marginals (propagate g)))]))
    [graph (or post (zipmap (keys priors) (map (fn [id] {:cpm (mapv P (:value (i (nodes id))))}) (vals priors))))] data))

(defn updated-variables [{:keys [fg updated marginals priors data] :as model}]
  (let [[g m]
          (last
           (update-variables
             (or updated (exp->fg :sp :tensor fg)) marginals priors data))]
    (-> model
      (assoc :marginals m)
      (assoc :updated g))))