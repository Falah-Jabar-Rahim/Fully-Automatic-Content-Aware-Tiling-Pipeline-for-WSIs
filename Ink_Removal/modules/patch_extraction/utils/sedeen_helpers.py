from typing import Dict, List, Union, Any
import xml.etree.ElementTree as ET

from shapely.geometry import Point,Polygon
import numpy as np

class Labels:
    """
    Class handling labels. It can handle any type of attributes associated with the input labels.
    This class main function is to fetch index based on any of the fields

    Assumes input is list of dictionary of form [{field1:,value:,field2:}]
    """
    def __init__(
        self,
        labels:List[Dict],
    )->None:
        self.labels = labels

        self._initialize_labels()

    def _initialize_labels(self):
        #Initialize all the keys
        for key in self.labels[0].keys():
            setattr(self,key,{})
        
        self._populate_fields()
    
    def _populate_fields(self)->None:
        for i in range(len(self.labels)):
            for key,value in self.labels[i].items():
                getattr(self,key)[i] = self._tolower(value)

    def get_field(self,key:str)->Dict:
        """
        Fetches field given attribute key
        """
        return getattr(self,key)

    def get_label(self,key:str,value:Any)->Dict:
        """
        Given a key and value fetches label
        """
        fields = self.get_field(key)
        label_key = list(fields.keys())[list(fields.values()).index(self._tolower(value))]
        
        label = {}
        for key in self.labels[0].keys():
             #To account for any modifications
             label[key] = getattr(self,key)[label_key]
        return label
    
    def modify_label(self,pairs:Dict,field_key:str,field_value:str)->None:
        """
        Based on input dictionary with key and value, changes the current key with new key
        Mainly for modifiying class label based on class name. Default class label is index+1
        Parameters:
            pairs: Dictionary with pairing of value from attribute field_key and value from attributre field_value
                    for ex pairs = {'value from field_key attribute':'value from field_value attribute'}
            field_key: attribute name of key of pairs dictionary
            field_value: attribute name of value of pairs dictionary
        """
        att_value = self.get_field(field_value)

        #Get current key values and replace with new key values
        for keys,value in pairs.items():
            current_key = list(att_value.keys())[list(att_value.values()).index(self._tolower(value))]
            getattr(self,field_key)[current_key] =  self._tolower(keys)
    
    @staticmethod
    def _tolower(value:Any)->Union[int,str]:
        if isinstance(value,str):
            value = value.lower()
        return value

    def __str__(self) -> str:
        result = ""
        for key in self.labels[0].keys():
            result +=  key + ": " +  str(list(getattr(self,key,{}).values())) + "\n"
        return result

class Annotation:
    def __init__(self, type: Union[Point,Polygon], index: int, label: Labels, coordinates:np.array, **geo_kwargs)->None:
        self._type = type
        self._index = index
        self._label = label
        self._coordinates = coordinates
        self._kwargs = geo_kwargs

        for key, value in geo_kwargs.items():
            setattr(self, key, value)

        self._geometry = self._type(coordinates,**self._kwargs)

    @property
    def geometry(self) -> Union[Point,Polygon]:
        return self._geometry
    
    @property
    def type(self) ->  Union[Point,Polygon]:
        return self._type
    
    @property
    def index(self) -> int:
        return self._index

    @property
    def label(self) -> Labels:
        return self._label

    @property
    def coordinates(self):
        return self._coordinates

    def todict(self):
        return dict(
            index=self.index,
            coordinates=self.coordinates.tolist(),
            label=self.label.todict(),
            **self._kwargs
        )

    def __str__(self):
        return ",".join(map(str, list(self.todict().values())))

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other)->bool:
        return (
            (self.geometry == other.geometry)
            and (self.label == other.label)
        )
