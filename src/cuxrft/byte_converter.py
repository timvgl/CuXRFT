
import re

def getValueAsClass(floatUnitTuple):
    if (floatUnitTuple[0][1] == 'B'):
        return Bit(float(floatUnitTuple[0][0]))
    elif (floatUnitTuple[0][1] == 'Ni'):
        return Nibble(float(floatUnitTuple[0][0]))
    elif (floatUnitTuple[0][1] == 'KB'):
        return Kilobyte(float(floatUnitTuple[0][0]))
    elif (floatUnitTuple[0][1] == 'KiB'):
        return Kibibyte(float(floatUnitTuple[0][0]))
    elif (floatUnitTuple[0][1] == 'MB'):
        return Megabyte(float(floatUnitTuple[0][0]))
    elif (floatUnitTuple[0][1] == 'MiB'):
        return Mebibyte(float(floatUnitTuple[0][0]))
    elif (floatUnitTuple[0][1] == 'GB'):
        return Gigabyte(float(floatUnitTuple[0][0]))
    elif (floatUnitTuple[0][1] == 'GiB'):
        return Gibibyte(float(floatUnitTuple[0][0]))

def convertToBytesByUnit(valueWithUnit: str) -> float:
    pattern = r"(-?\d+(?:\.\d+)?)([a-zA-Z]+)"
    floatUnitTuple = re.findall(pattern, valueWithUnit.replace(' ', ''))
    return getValueAsClass(floatUnitTuple).to_bytes()

def convertToBitsByUnit(valueWithUnit: str) -> float:
    pattern = r"(-?\d+(?:\.\d+)?)([a-zA-Z]+)"
    floatUnitTuple = re.findall(pattern, valueWithUnit.replace(' ', ''))
    return getValueAsClass(floatUnitTuple).to_bits()

class Bit():
    def __init__(self, value: float):
        self.value = value
    
    def to_bytes(self):
        return self.value / 8
    
    def to_bits(self):
        return self.to_bytes()*8
    
class Nibble():
    def __init__(self, value: float):
        self.value = value
    
    def to_bytes(self):
        return self.value / 2
    
    def to_bits(self):
        return self.to_bytes()*8

class Kilobyte():
    def __init__(self, value: float):
        self.value = value
    
    def to_bytes(self):
        return self.value * 1000
    
    def to_bits(self):
        return self.to_bytes()*8
    
class Kibibyte():
    def __init__(self, value: float):
        self.value = value
    
    def to_bytes(self):
        return self.value * 1024
    
    def to_bits(self):
        return self.to_bytes()*8
    
class Megabyte():
    def __init__(self, value: float):
        self.value = value
    
    def to_bytes(self):
        return Kilobyte(self.value).to_bytes() * 1000
    
    def to_bits(self):
        return self.to_bytes()*8

class Mebibyte():
    def __init__(self, value: float):
        self.value = value
    
    def to_bytes(self):
        return Kibibyte(self.value).to_bytes() * 1024
    
    def to_bits(self):
        return self.to_bytes()*8
    
class Gigabyte():
    def __init__(self, value: float):
        self.value = value
    
    def to_bytes(self):
        return Megabyte(self.value).to_bytes() * 1000
    
    def to_bits(self):
        return self.to_bytes()*8
    
class Gibibyte():
    def __init__(self, value: float):
        self.value = value
    
    def to_bytes(self):
        return Mebibyte(self.value).to_bytes() * 1024
    
    def to_bits(self):
        return self.to_bytes()*8

    