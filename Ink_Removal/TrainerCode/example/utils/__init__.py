'''
 file_name : __init__.py
 For automatically importing your classes in the folder where the file __init__.py is situated
'''
import os
from inspect import isclass
from pathlib import Path
from importlib import import_module

# iterate through the modules in the current package
package_dir = Path(__file__).resolve().parent

def get_modules(folder_name, parent_module):
    for name in os.listdir(folder_name):
        if (name.endswith(".py")) and (name!=Path(__file__).name):
            # import the module and iterate through its attributes
            module_name = name[:-3]
            module = import_module(f"{parent_module}.{module_name}")
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)

                if isclass(attribute):
                    # Add the class to this package's variables
                    globals()[attribute_name] = attribute
        elif (folder_name / name).is_dir():
            get_modules(folder_name / name, parent_module + "." + name)


get_modules(package_dir,package_dir.name)