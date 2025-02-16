# Note there is some still issue with holes. Yet to rectify. Creating wrong pairs and with 0 area


from typing import Dict, List, Union
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Point, Polygon

from .sedeen_helpers import Labels, Annotation

class SedeenAnnotationParser:
    """
    Class defined for handling annotations defined by Sedeen Viewer. The class converts the annotation to shapely objects 
    and handles the given labels by parsing xml files.
    The class handles annular annotations. However, to avoid lengthy computations, it is expected that the user will provide
    color used for Annular annotations.
    The class doesn't handle pointset well and filters them out
    """
    TYPES = {
        "polygon": Polygon,
        "rectangle": Polygon,
        "dot": Point,
        "spline": Polygon,
        "pointset": Point,
        "ellipse": Polygon,
        "polyline": Polygon
    }

    def __init__(self, renamed_label:Dict=None, annular_color:str=None) -> None:
        """
        Parameters:
            renamed_label: For assigning class names to different colors, to be input as {'color1':'name1'}
            annular_color: Color which is associated with annular annotations
        """
        self._renamed_label = renamed_label
        
        self.annular_label = annular_color
        self.hollow_label = {"name":"rest","value":0,"color":self.annular_label}

    def get_available_labels(self,open_annotations):
        labels = []
        for child in open_annotations:
            if (child.tag == "graphic") & (child.attrib.get("type")!="text"):
                for grandchild in child:
                    if grandchild.tag == "pen":
                        labels.append(grandchild.attrib.get("color"))
        #Construct using Labels
        labels_construct = []
        labels = list(set(labels))
        renamed_list = [i for i in self._renamed_label.values()]
        for i in range(len(labels)):
            #Only include labels in renamed_label
            if (labels[i] in renamed_list):
                name = labels[i]
                labels_construct.append({"name":name,"value":i+1,"color":labels[i]})
                
        #Check if all labels are available as given in the new set of labels, if not add the new label
        att_value_list = [i["color"] for i in labels_construct]
        count = len(att_value_list)
        for keys,values in self._renamed_label.items():
            if values not in att_value_list:
               temp_dict = {}
               count+=1
               temp_dict["value"] = count
               temp_dict["name"] = values
               temp_dict["color"] = values
               labels_construct.append(temp_dict)

        labels_final = Labels(labels_construct)
        labels_final = self._modify_labelset(labels_final)
        return labels_final

    def _get_label(self, child, labels: Labels, type):
        name = self._get_label_name(child, labels, type)
        if name not in list(labels.get_field("color").values()):
            return None

        label = labels.get_label("color",name)
        return label

    @staticmethod
    def _get_label_name(child, labels, type) -> str:
        return child.attrib.get("color")

    def _modify_labelset(self,labels:Labels):
        #Rename labels
        labels.modify_label(self._renamed_label,"name","color")
        #Reorder labels according to name
        temp_pair = {}
        for i,values in enumerate(self._renamed_label.keys()):
            temp_pair[i+1] = values.lower()
        labels.modify_label(temp_pair,"value","name")
        return labels

    @staticmethod
    def _get_annotation_type(child):
        annotation_type = child.attrib.get("type").lower()
        if annotation_type in SedeenAnnotationParser.TYPES:
            return SedeenAnnotationParser.TYPES[annotation_type]
        raise ValueError(f"unsupported annotation type in {child}")

    @staticmethod
    def _get_coords(child):
        coords = []
        for coordinates in child:
            nums = coordinates.text.split(",")
            coords.append([float(nums[0]),float(nums[1])])
        return coords

    @staticmethod
    def _create_new_annotation(index:int,type:Union[Point,Polygon],coords:np.array,label:Labels,holes:List=[]):
        annotation = {"index":index,
                      "type":type,
                      "coordinates":coords,
                      "label":label}
        if len(holes)!=0:
            annotation["holes"] = holes
        return Annotation(**annotation)  
    
    def _modify_annotations(self,annotations,annular_index):
        # annular_annotations = [annotations[idx] for idx in annular_index]
        area_annotation = np.array([annotations[idx].geometry.area for idx in annular_index])
        index = list(np.array(annular_index)[np.argsort(area_annotation)[::-1]])
        index_stack = index.copy()

        while len(index_stack)!=0:
            idx_i = index_stack.pop(0)
            for j,idx_j in enumerate(index_stack):
                
                if annotations[idx_i].geometry.buffer(0).contains(annotations[idx_j].geometry.buffer(0)):
                    index_stack.pop(j)
                    # modify the annotations
                    annotations[idx_i] = self._create_new_annotation(index = idx_i,
                                                                    type = annotations[idx_i].type,
                                                                    coords = annotations[idx_i].coordinates,
                                                                    label = annotations[idx_i].label,
                                                                    holes = [annotations[idx_j].coordinates])

                    annotations[idx_j] = self._create_new_annotation(index = idx_j,
                                                                    type = annotations[idx_j].type,
                                                                    coords = annotations[idx_j].coordinates,
                                                                    label = self.hollow_label)
                    break  
        return annotations            


    def _parse(self, path):
        tree = ET.parse(path)
        annot = tree.getroot()
        for parent in annot:
            for child in parent:
                if child.tag=="overlays":
                    open_annot = child
                    break

        labels = self.get_available_labels(open_annot)

        for child in open_annot:
            if (child.tag == "graphic") & (child.attrib.get("type")!="text"):
                type = self._get_annotation_type(child)
                for grandchild in child:
                    if grandchild.tag == "pen":
                        label = self._get_label(grandchild,labels,type)
                    elif grandchild.tag == "point-list":
                        coordinates = self._get_coords(grandchild)
                        if len(coordinates)>0 and (label is not None):
                            yield {
                                    "type": type,
                                    "coordinates": coordinates,
                                    "label": label,
                                }

    def parse(self, path) -> List[Annotation]:

        if not Path(path).is_file():
            raise FileNotFoundError(path)

        annotations = []
        ann_objects = []
        annular_index = []
        index = 0
        for annotation in self._parse(path):
            annotation["index"] = index
            annotation["coordinates"] = np.array(annotation["coordinates"])
            #note the index of the potentially annular annotations, in particular green        
            label_name = annotation["label"]
            if ((annotation["type"]==Polygon) and (len(annotation["coordinates"])<3)):
                continue
            temp_annotation = Annotation(**annotation)
            #necessary step due to number of repeated annotations
            ann_objects.append(temp_annotation)
            if (not (isinstance(temp_annotation.type,Point))) and (temp_annotation.geometry.area!=0) and (temp_annotation not in annotations):
                annotations.append(temp_annotation)
                if label_name["color"] == self.annular_label:
                    annular_index.append(index)
                index+=1

        #Add holes in annular annotations
        annotations = self._modify_annotations(annotations,annular_index)

        return annotations