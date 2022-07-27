# Copyright 2022 The Quket Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# limitations under the License.
"""
#######################
#        quket        #
#######################

memory.py

Utilities.

"""
import psutil

def convert_memory_unit(val, unit):
    if type(unit) is str:
        if unit.lower() == 'kb':
            return val/(1024**1) 
        elif unit.lower() == 'mb':
            return val/(1024**2) 
        elif unit.lower() == 'gb':
            return val/(1024**3) 
    else:
        raise TypeError(f"unit = {unit} is not str. Use either 'kb', 'mb', 'gb'.")

def available(unit=None):
    val = psutil.virtual_memory().available
    if unit is None:
        return val
    else:
        return convert_memory_unit(val, unit)

def used(unit=None):
    val = psutil.virtual_memory().used
    if unit is None:
        return val
    else:
        return convert_memory_unit(val, unit)

def total(unit=None):
    val = psutil.virtual_memory().total
    if unit is None:
        return val
    else:
        return convert_memory_unit(val, unit)

def percent(unit=None):
    val = psutil.virtual_memory().percent
    if unit is None:
        return val
    else:
        return convert_memory_unit(val, unit)

