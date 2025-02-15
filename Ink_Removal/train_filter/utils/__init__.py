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
for name in os.listdir(package_dir):
    if (name.endswith(".py")) and (name!=Path(__file__).name):
        # import the module and iterate through its attributes
        module_name = name[:-3]
        module = import_module(f"{__name__}.{module_name}")
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)

            if isclass(attribute):
                # Add the class to this package's variables
                globals()[attribute_name] = attribute