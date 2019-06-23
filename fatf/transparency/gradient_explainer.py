"""
The :mod:`fatf.interpretability.gradient_explainer` module holds the object and
functions relevant to performing counterfactual explanations.
"""

# Author: Rafael Poyiadzi <rp13102@bristol.ac.uk>
# License: BSD 3 clause

import numpy as np
from typing import Optional, Callable, List, Dict
import matplotlib.pyplot as plt
from itertools import cycle

class Explainer2(object):
    """Class for doing counterfactual explanations in the provided dataset.

    """
    def __init__(self,
                 model: Callable,
                 threshold: Optional[float] = None,
                 boundaries: Optional[Dict] = None,
                 ):

        # for estimating the gradient
        self.step = 1e-7
        
        # for gradient descent
        self.step_size = 1e-2
        
        self.last_diff = 10000

        self.n_classes = model.classes_.shape[0]
        self.model = model
        if not threshold:
            self.threshold = 0.05
        else:
            self.threshold = threshold
        self.boundaries = boundaries
        self.counter = 0
        self.stagnation_bound = 10

    def __estimate_gradient(self,
                            instance: np.array):
        idx = self.counter
        instance[idx] += self.step
        eval_up = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class]
        instance[idx] -= 2 * self.step
        eval_down = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class]
        instance[idx] += self.step
        grad_fx = (eval_up - eval_down) / (2 * self.step)

        return grad_fx.reshape(-1, 1)

    def __eval_condition(self,
                         test_prediction: np.array) -> bool:
        diff = np.abs(1 - test_prediction)
        self.last_diff = diff
        return diff < self.threshold

    def __optimise(self,
                   instance: np.array,
                   ) -> np.array:
        self.st = 0
        satisfied = self.__eval_condition(self.current_prediction)
        stagnated = 0
        while not satisfied:
            self.counter += 1
            self.counter = self.counter%self.n_ftrs
            if self.counter in self.nottochange:
                continue
            self.st += 1
            grad_fx = self.__estimate_gradient(instance)
            grad_fx *= self.step_size
            if self.boundaries is not None:
                newx_i = instance[self.counter] + grad_fx
                if (newx_i >= self.boundaries[self.counter]['min'] and
                    newx_i <= self.boundaries[self.counter]['max']):
                    instance[self.counter] = newx_i
                    stagnated = 0
                    pred = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class]
                    satisfied = self.__eval_condition(pred)
                else:
                    stagnated += 1
                    if stagnated == self.stagnation_bound:
                        return instance
            else:
                instance[self.counter] += grad_fx
                pred = self.model.predict_proba(instance.reshape(1, -1))[0][self.target_class]
                satisfied = self.__eval_condition(pred)
            n_steps = 1000
            colors=cycle(list((plt.cm.rainbow(np.linspace(0,1,n_steps)))))

            plt.scatter(instance[0], instance[1], color=next(colors), marker='x')
        return instance

    def explain(self,
                subject_instance: np.array,
                target_class: int,
                nottochange: Optional[List[int]] = None):
        self.subject_instance = subject_instance
        self.target_class = target_class
        self.n_ftrs = len(self.subject_instance)
        self.current_prediction = self.model.predict_proba(self.subject_instance.reshape(1, -1))[0][self.target_class]
        if not nottochange:
            self.nottochange = []
            self.tosamplefrom = self.n_ftrs
        else:
            self.nottochange = nottochange
            self.tosamplefrom = list(range(self.n_ftrs))[:]
            for item in self.nottochange:
                self.tosamplefrom.pop(self.tosamplefrom.index(item))

        instance = self.subject_instance.copy(order='K')
        return self.__optimise(instance)
