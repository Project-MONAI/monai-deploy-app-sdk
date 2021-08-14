from .factory import ModelFactory
from .model import Model
from .torch_model import TorchScriptModel
from .named_model import NamedModel
from .triton_model import TritonModel

Model.register([TritonModel, NamedModel, TorchScriptModel, Model])
