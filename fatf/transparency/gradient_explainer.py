"""
The :mod:`fatf.interpretability.gradient_explainer` module holds the object and
functions relevant to performing counterfactual explanations.
"""

# Author: Rafael Poyiadzi <rp13102@bristol.ac.uk>
# License: BSD 3 clause

import numpy as np
from typing import Optional, Union, Callable, List, Dict
from fatf.utils.validation import check_array_type
import matplotlib.pyplot as plt
from itertools import cycle

class Explainer2(object):
    """Class for doing counterfactual explanations in the provided dataset.

    """
    def __init__(self,
                 model: Callable,
                 reg: Optional[Union[float, int]] = None,
                 threshold: Optional[float] = None,
                 boundaries: Optional[Dict] = None,
                 cost_func: Optional[Callable] = None):
        
        self.step = 0.00001
        self.cycle = True
        self.last_diff = 10000
        if reg is not None:
            self.reg = [reg]
        else:
            self.reg = [5, 10, 20]
        
        self.model = model
        if not threshold:
            self.threshold = 0.05
        else:
            self.threshold = threshold
        self.boundaries = boundaries
        self.counter = 0
        self.stagnation_bound = 10
        
        self.own_func = False
        if cost_func is not None:
            self.cost_func = cost_func
            self.own_func = True
    
    def __estimate_gradient(self, 
                            instance: np.array):
        idx = self.counter
        instance[idx] += self.step
        eval_up = self.model.predict_proba(instance.reshape(1, -1))
        instance[idx] -= 2 * self.step
        eval_down = self.model.predict_proba(instance.reshape(1, -1))
        instance[idx] += self.step
        grad_fx = (eval_up - eval_down) / (2 * self.step)

        return grad_fx.reshape(-1, 1)
    
    def __eval_condition(self, 
                         test_prediction: np.array) -> bool:
        diff = np.linalg.norm(self.target_prediction - test_prediction)
# =============================================================================
#         if diff > self.last_diff + 0.2:
#             return True
# =============================================================================
        self.last_diff = diff
        return diff < self.threshold
    
    def __optimise(self,
                   instance: np.array,
                   reg: int) -> np.array:
        self.st = 0
        satisfied = self.__eval_condition(self.current_prediction)
        stagnated = 0
        while not satisfied:
            if self.cycle:
                self.counter += 1
                self.counter = self.counter%self.n_ftrs
                if self.counter in self.nottochange:                    
                    continue
            else:
                self.counter = np.random.choice(self.tosamplefrom) 
            self.st += 1
            grad_fx = self.__estimate_gradient(instance)
            part0 = np.linalg.norm(grad_fx)**2
            part1a = self.model.predict_proba(instance.reshape(1, -1)).reshape(-1, 1)
            part1b = self.target_prediction - part1a
            part1 = np.dot(part1b.T, grad_fx)
            delta = part1 / (part0 + reg)
            if self.boundaries is not None:
                newx_i = instance[self.counter] + 2 * delta
                if (newx_i >= self.boundaries[self.counter]['min'] and
                    newx_i <= self.boundaries[self.counter]['max']):
                    instance[self.counter] = newx_i
                    stagnated = 0
                    part1a = self.model.predict_proba(instance.reshape(1, -1)).reshape(-1, 1)
                    satisfied = self.__eval_condition(part1a)
                else:
                    stagnated += 1
                    if stagnated == self.stagnation_bound:
                        return instance
            else:
                instance[self.counter] += 2 * delta  
                satisfied = self.__eval_condition(part1a)
            n_steps = 1000
            colors=cycle(list((plt.cm.rainbow(np.linspace(0,1,n_steps)))))

            plt.scatter(instance[0], instance[1], color=next(colors), marker='x')
        return instance
    
    def explain(self,
                subject_instance: np.array,
                target_prediction: np.array,
                nottochange: Optional[List[int]] = None):
        self.subject_instance = subject_instance
        self.target_prediction = target_prediction.reshape(-1, 1)
        self.n_classes = self.target_prediction.shape[0]
        self.n_ftrs = len(self.subject_instance)
        self.current_prediction = self.model.predict_proba(
                                    self.subject_instance.reshape(1, -1))
        
        if not nottochange:
            self.nottochange = []
            self.tosamplefrom = self.n_ftrs
        else:
            self.nottochange = nottochange
            self.tosamplefrom = list(range(self.n_ftrs))[:]
            for item in self.nottochange:
                self.tosamplefrom.pop(self.tosamplefrom.index(item))
        
        instance = self.subject_instance.copy(order='K')
        previous_modified_x = 0
        for reg in self.reg:
            print(reg)
            modified_x = self.__optimise(instance, reg)
            
            pred = self.model.predict(modified_x.reshape(1, -1))
            print(modified_x, pred)
            if pred != np.argmax(self.target_prediction):
                return previous_modified_x
            else:
                previous_modified_x = modified_x
            print(previous_modified_x)
            print('\n')
        return modified_x


