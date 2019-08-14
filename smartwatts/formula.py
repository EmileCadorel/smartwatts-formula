# Copyright (C) 2018  INRIA
# Copyright (C) 2018  University of Lille
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import hashlib
import pickle
import warnings
from collections import OrderedDict
from typing import List, Dict, Union

from scipy.linalg import LinAlgWarning
from sklearn.linear_model import Ridge

from smartwatts.topology import CPUTopology

# make scikit-learn more silent
warnings.filterwarnings('ignore', category=LinAlgWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class PowerModelNotInitializedException(Exception):
    """
    This exception happens when a user try to compute a power estimation without having learned a power model.
    """
    pass


class Sample:
    """
    This class stores the events and allows some operations on them.
    """

    def __init__(self, events: Dict[str, int]):
        """
        Initialize a new events wrapper.
        """
        self.events = events

    def values(self) -> List[int]:
        """
        Creates and return a list of events value from the Core events group.
        :return: List containing the events value sorted by event name.
        """
        return [v for _, v in sorted(self.events.items())]


class SamplesContainer:
    """
    This container stores the samples to be used for learning a power model.
    """

    def __init__(self) -> None:
        """
        Initialize the reports container.
        """
        self.X = []
        self.y = []

    def store(self, ref_power: float, sample: Sample) -> None:
        """
        Store the values contained in the given report.
        """
        self.X.append(sample.values())
        self.y.append(ref_power)

    def __len__(self) -> int:
        return len(self.X)


class PowerModel:
    """
    This Power model compute the power estimations and handle the learning of a new model when needed.
    """

    def __init__(self, frequency: int) -> None:
        """
        Initialize a new power model.
        :param frequency: Frequency of the power model
        """
        self.frequency = frequency
        self.model: Union[Ridge, None] = None
        self.hash: str = 'uninitialized'
        self.id = 0
        self.samples: SamplesContainer = SamplesContainer()

    @staticmethod
    def _select_best_model(first_model: Ridge, second_model: Ridge, sample: Sample, ref_power: float) -> Ridge:
        """
        Compare the two given models using the provided sample.
        :param first: First power model
        :param second: Second power model
        :param sample: Sample to use to compare the two models
        :param ref_power: Reference global power measurement (RAPL)
        :return: The model having the lowest error for the given sample
        """
        if first_model is None:
            return second_model

        first_predict = first_model.predict([sample.values()]).item(0)
        first_error = abs(ref_power - first_predict)

        second_predict = second_model.predict([sample.values()]).item(0)
        second_error = abs(ref_power - second_predict)

        if first_error > second_error:
            return second_model

        return first_model

    def learn(self, ref_power: float, global_core: Dict[str, int]):
        """
        Learn a new power model using the stored reports and update the formula hash.
        """
        sample = Sample(global_core)
        self.samples.store(ref_power, sample)
        if len(self.samples) < 3:
            return

        new_model = Ridge().fit(self.samples.X, self.samples.y)
        self.model = self._select_best_model(self.model, new_model, sample, ref_power)

        if self.model == new_model:
            self.id += 1
            self.hash = hashlib.blake2b(pickle.dumps(self.model), digest_size=20).hexdigest()

    def compute_global_power_estimation(self, global_core: Dict[str, int]) -> float:
        """
        Compute the global power estimation using the power model.
        :param global_core: Core events group of all targets
        :return: Power estimation of all running targets using the power model
        """
        if not self.model:
            raise PowerModelNotInitializedException()

        events = Sample(global_core)
        return self.model.predict([events.values()]).item(0)

    def compute_target_power_estimation(self, ref_power: float, global_core: Dict[str, int], target_core: Dict[str, int]) -> (float, float):
        """
        Compute a power estimation for the given target.
        :param ref_power: Reference global power measurement (RAPL)
        :param global_core: Core events group of all targets
        :param target_core: Core events group of any target
        :return: Power estimation for the given target and ratio of the target on the global power consumption
        :raise: PowerModelNotInitializedException when the power model is not initialized
        """
        if not self.model:
            raise PowerModelNotInitializedException()

        system = Sample(global_core).values()
        target = Sample(target_core).values()

        sum_coefs = sum(self.model.coef_)
        ratio = 0.0
        for index, coef in enumerate(self.model.coef_):
            try:
                ratio += (coef / sum_coefs) * (target[index] / system[index])
            except ZeroDivisionError:
                pass

        target_power = ref_power * ratio
        if target_power < 0.0:
            return 0.0, 0.0

        return target_power, ratio


class SmartWattsFormula:
    """
    This formula compute per-target power estimations using hardware performance counters.
    """

    def __init__(self, cpu_topology: CPUTopology) -> None:
        """
        Initialize a new formula.
        :param cpu_topology: CPU topology to use
        """
        self.cpu_topology = cpu_topology
        self.models = self._gen_models_dict()

    def _gen_models_dict(self) -> Dict[int, PowerModel]:
        """
        Generate and returns a layered container to store per-frequency power models.
        :return: Initialized Ordered dict containing a power model for each frequency layer.
        """
        return OrderedDict((freq, PowerModel(freq)) for freq in self.cpu_topology.get_supported_frequencies())

    def _get_frequency_layer(self, frequency: float) -> int:
        """
        Find and returns the nearest frequency layer for the given frequency.
        :param frequency: CPU frequency
        :return: Nearest frequency layer for the given frequency
        """
        last_layer_freq = 0
        for current_layer_freq in self.models.keys():
            if frequency < current_layer_freq:
                return last_layer_freq
            last_layer_freq = current_layer_freq

        return last_layer_freq

    def compute_pkg_frequency(self, system_msr: Dict[str, int]) -> float:
        """
        Compute the average package frequency.
        :param system_msr: MSR events group of System target
        :return: Average frequency of the Package
        """
        return (self.cpu_topology.get_base_frequency() * system_msr['APERF']) / system_msr['MPERF']

    def get_power_model(self, system_core: Dict[str, int]) -> PowerModel:
        """
        Fetch the suitable power model for the current frequency.
        :param system_core: Core events group of System target
        :return: Power model to use for the current frequency
        """
        return self.models[self._get_frequency_layer(self.compute_pkg_frequency(system_core))]
