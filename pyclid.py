import textwrap
from .veriloggen import *

from typing import Any, List, Dict

from enum import Enum
import logging


class UclidContext():
    # dict:
    #   modulename : str -> UclidModule
    modules             = {}
    # name of module in current context
    curr_module_name    = "main"

    @staticmethod
    def clearAll():
        """
            Deletes all context information (restart).
        """
        UclidContext.curr_module_name   = "main"
        UclidContext.modules            = { "main" : UclidModule("main") }

    @staticmethod
    def setContext(module_name):
        UclidContext.curr_module_name = module_name
        if module_name not in UclidContext.modules:
            UclidContext.modules[module_name]  = UclidModule(module_name)

    @staticmethod
    def __inject__():
        acc = ""
        for modulename in UclidContext.modules:
            acc += UclidContext.modules[modulename].__inject__()
        return acc


setContext  = UclidContext.setContext
clearAll    = UclidContext.clearAll


class UclidElement():
    def __init__(self) -> None:
        pass
    def __inject__(self):
        raise NotImplementedError


class UclidModule(UclidElement):

    def __init__(self, _name, _init=None, _next=None, _control=None, _properties={}):
        super().__init__()
        # Module elements
        self.name       : str               = _name
        self.init       : UclidInit         = _init
        self.next       : UclidNext         = _next
        self.control    : UclidControlBlock = _control
        
        # Module declarations
        # dict: declname : str -> UclidDecl
        self.var_decls       : Dict[str, UclidDecl] = dict()
        # dict: declname : str -> UclidDecl
        self.const_decls     : Dict[str, UclidDecl] = dict()
        # dict: declname : str -> UclidDecl
        self.type_decls       : Dict[str, UclidDecl] = dict()
        # dict: declname : str -> UclidDecl
        self.instance_decls  : Dict[str, UclidDecl] = dict()
        # dict
        self.ip_var_decls    : Dict[str, UclidDecl] = dict() 
        # dict
        self.op_var_decls    : Dict[str, UclidDecl] = dict() 
        # list
        self.import_decls    : List[UclidImportDecl]    = [] 
        # list
        self.define_decls    : List[UclidDefine]        = [] 
        # list
        self.procedure_defns : List[UclidProcedureDecl] = [] 
        # list
        self.module_assumes  : List[UclidAxiomDecl]     = [] 
        # list
        self.module_properties  : List[UclidSpecDecl]   = []

        UclidContext.modules[self.name] = self

    def mkType(self, name, decl):
        if name in self.type_decls:
            logging.warn("Redeclaration of type named {} in module {}".format(name, self.name))
        else:
            self.type_decls[name] = decl

    def mkVar(self, name, decl):
        if name in self.var_decls or name in self.op_var_decls or name in self.ip_var_decls:
            logging.warn("Redeclaration of name {} in module {}".format(name, self.name))
        else:
            if decl.porttype == PortType.input:
                self.ip_var_decls[name] = decl
            elif decl.porttype == PortType.output:
                self.op_var_decls[name] = decl
            else:
                self.var_decls[name] = decl

    def mkConst(self, name, decl):
        if name in self.const_decls:
            logging.warn("Redeclaration of const {} in module {}".format(name, self.name))
        else:
            self.const_decls[name] = decl

    def mkInstance(self, name, decl):
        if name in self.instance_decls:
            logging.warn("Redeclaration of instance {}".format(name))
        else:
            self.instance_decls[name] = decl

    def mkImport(self, decl):
        self.import_decls.append(decl)

    def mkDefine(self, decl):
        self.define_decls.append(decl)

    def mkProcedure(self, defn):
        self.procedure_defns.append(defn)

    def mkAssume(self, assm):
        self.module_assumes.append(assm)

    def mkProperty(self, prop):
        self.module_properties.append(prop)

    def __type_decls__(self):
        return "\n".join([textwrap.indent(decl.__inject__(), '\t')
            for k, decl in self.type_decls.items()])

    def __var_decls__(self):
        return "\n".join([textwrap.indent(decl.__inject__(), '\t')
            for k, decl in self.var_decls.items()])

    def __const_decls__(self):
        return "\n".join([textwrap.indent(decl.__inject__(), '\t')
            for k, decl in self.const_decls.items()])

    def __instance_decls__(self):
        return "\n".join([textwrap.indent(decl.__inject__(), '\t')
            for k, decl in self.instance_decls.items()])

    def __import_decls__(self):
        return "\n".join([textwrap.indent(decl.__inject__(), "\t") for
            decl in self.import_decls])

    def __define_decls__(self):
        return "\n".join([decl.__inject__() for
            decl in self.define_decls])

    def __procedure_defns__(self):
        return "\n".join([textwrap.indent(defn.__inject__(), "\t") for
            defn in self.procedure_defns])

    def __module_assumes__(self):
        return "\n".join([textwrap.indent(assm.__inject__(), "\t") for
            assm in self.module_assumes])

    def __module_properties__(self):
        return "\n".join([textwrap.indent(assm.__inject__(), "\t") for
            assm in self.module_properties])


    def __inject__(self):
        # dependencies = [inst.obj.module.name for inst in UclidContext.instance_decls[self.name].values()]
        # acc = "\n\n".join([UclidContext.modules[dep].__inject__() for dep in dependencies])
        acc = ""
        init_code = textwrap.indent(self.init.__inject__() if self.init is not None else "", "\t")
        next_code = textwrap.indent(self.next.__inject__() if self.next is not None else "", "\t")
        control_code = textwrap.indent(self.control.__inject__() if self.control is not None else "", "\t")
        decls_code = textwrap.dedent("""
            \t// Imports
            {}

            \t// Types
            {}

            \t// Variables
            {}

            \t// Consts
            {}

            \t// Instances
            {}

            \t// Defines
            {}

            \t// Procedures
            {}

            \t// Assumes
            {}

            \t// Properties
            {}
            """).format(
                self.__import_decls__(),
                self.__type_decls__(),
                self.__var_decls__(),
                self.__const_decls__(),
                self.__instance_decls__(),
                self.__define_decls__(),
                self.__procedure_defns__(),
                self.__module_assumes__(),
                self.__module_properties__()
            )
        acc += textwrap.dedent("""
            module {} {{
            {}

            {}

            {}
            }}
            """).format(self.name, decls_code, init_code, next_code, control_code)
        return acc


class UclidInit(UclidElement):
    def __init__(self, block) -> None:
        super().__init__()
        if isinstance(block, UclidBlockStmt):
            self.block = block
        elif isinstance(block, list):
            self.block = UclidBlockStmt(block, 1)
        elif isinstance(block, UclidStmt):
            self.block = UclidBlockStmt([block], 1)
        else:
            logging.error("Unsupported type {} of block in UclidInit".format(
                type(block)
            ))

    def __inject__(self):
        return textwrap.dedent("""
            init {{
            {}
            }}""").format(textwrap.indent(self.block.__inject__().strip(), '\t'))


class UclidNext(UclidElement):
    def __init__(self, block) -> None:
        super().__init__()
        if isinstance(block, UclidBlockStmt):
            self.block = block
        elif isinstance(block, list):
            self.block = UclidBlockStmt(block, 1)
        elif isinstance(block, UclidStmt):
            self.block = UclidBlockStmt([block], 1)
        else:
            logging.error("Unsupported type {} of block in UclidNext".format(
                type(block)
            ))

    def __inject__(self):
        return textwrap.dedent("""
            next {{
            {}
            }}""").format(textwrap.indent(self.block.__inject__().strip(), '\t'))

class Operators:
    """
    Comprehensive operator list
    """
    OpMapping = {
        "add"   : ("+",     2),
        "sub"   : ("-",     2),
        "umin"  : ("-",     1),
        "gt"    : (">",     2),
        "gte"   : (">=",    2),
        "lt"    : ("<",     2),
        "lte"   : ("<=",    2),
        "eq"    : ("==",    2),
        "neq"   : ("!=",    2),
        "not"   : ("!",     1),
        "xor"   : ("^",     2),
        "and"   : ("&&",   -1),
        "or"    : ("||",   -1),
        "implies"       : ("==>",   2),
        "bvadd" : ("+",     2),
        "bvsub" : ("-",     2),
        "bvand" : ("&",     2),
        "bvor"  : ("|",     2),
        "bvnot" : ("~",     1),
        "bvlt"  : ("<",     2),
        "bvult" : ("<_u",   2),
        "bvgt"  : (">",     2),
        "bvugt" : (">_u",   2),
        "next"          : ("X",     1),
        "eventually"    : ("F",     1),
        "always"        : ("G",     1),
        "concat" : ("++", -1)
    }

    def __init__(self, op_) -> None:
        self.op = op_
    def __inject__(self):
        return Operators.OpMapping[self.op][0]
    def __to__vlog__(self):
        return OpMapping[self.op][0]

# Base class for Uclid expressions
class UclidExpr(UclidElement):
    def __inject__(self):
        raise NotImplementedError
    def __to__vlog__(self, prefix=""):
        raise NotImplementedError
    def __make_op__(self, op, other):
        if isinstance(other, UclidExpr):
            return UclidOpExpr(op, [self, other])
        elif isinstance(other, int):
            return UclidOpExpr(op, [self, UclidLiteral(str(other))])
        elif isinstance(other, str):
            return UclidOpExpr(op, [self, UclidLiteral(other)])
        else:
            logging.error("Unsupported types for operation {}: {} and {}".format(
                op, 'UclidExpr', type(other)))

    def __eq__(self, other):
        return self.__make_op__("eq", other)
    def __ne__(self, other):
        return self.__make_op__("neq", other)
    def __add__(self, other):
        return self.__make_op__("add", other)
    def __invert__(self):
        return UclidOpExpr("not", [self])
    def __and__(self, other):
        if isinstance(other, UclidExpr):
            return UclidOpExpr("and", [self, other])
        else:
            logging.error("Unsupported types for operation {}: {} and {}".format(
                '&', 'UclidExpr', type(other)))


# All kinds of operators
class UclidOpExpr(UclidExpr):
    def __init__(self, op, children) -> None:
        super().__init__()
        self.op = op
        self.children = [UclidLiteral(str(child)) if isinstance(child, int) else child for child in children]
    def __inject__(self):
        children_code = ["({})".format(child.__inject__()) for child in self.children]
        oprep = Operators.OpMapping[self.op]
        if oprep[1] == 1:
            assert len(children_code) == 1, "Unary operator must have one argument"
            return "{} {}".format(oprep[0], children_code[0])
        if oprep[1] == 2:
            assert len(children_code) == 2, "Binary operator must have two arguments"
            return "{} {} {}".format(children_code[0], oprep[0], children_code[1])
        if oprep[1] == -1:
            return (" "+oprep[0]+" ").join(children_code)
        else:
            logging.error("Operator arity not yet supported")
    def __to__vlog__(self, prefix=""):
        return VOpExpr(self.op, [child.__to__vlog__(prefix) for child in self.children])

def Uadd(args):
    return UclidOpExpr("add", args)
def Usub(args):
    return UclidOpExpr("sub", args)
def Uumin(args):
    return UclidOpExpr("umin", args)
def Ugt(args):
    return UclidOpExpr("gt", args)
def Ugte(args):
    return UclidOpExpr("gte", args)
def Ult(args):
    return UclidOpExpr("lt", args)
def Ulte(args):
    return UclidOpExpr("lte", args)
def Ueq(args):
    return UclidOpExpr("eq", args)
def Uneq(args):
    return UclidOpExpr("neq", args)
def Unot(args):
    return UclidOpExpr("not", args)
def Uxor(args):
    return UclidOpExpr("xor", args)
def Uand(args):
    return UclidOpExpr("and", args)
def Uor(args):
    return UclidOpExpr("or", args)
def Uimplies(args):
    return UclidOpExpr("implies", args)
def Ubvadd(args):
    return UclidOpExpr("bvadd", args)
def Ubvsub(args):
    return UclidOpExpr("bvsub", args)
def Ubvand(args):
    return UclidOpExpr("bvand", args)
def Ubvor(args):
    return UclidOpExpr("bvor", args)
def Ubvnot(args):
    return UclidOpExpr("bvnot", args)
def Ubvlt(args):
    return UclidOpExpr("bvlt", args)
def Ubvult(args):
    return UclidOpExpr("bvult", args)
def Ubvgt(args):
    return UclidOpExpr("bvgt", args)
def Ubvugt(args):
    return UclidOpExpr("bvugt", args)
def Unext(args):
    return UclidOpExpr("next", args)
def Ueventually(args):
    return UclidOpExpr("eventually", args)
def Ualways(args):
    return UclidOpExpr("always", args)
def Uconcat(args):
    return UclidOpExpr("concat", args)

class UclidBVSignExtend(UclidExpr):
    def __init__(self, var, ewidth):
        super().__init__()
        self.var = var
        self.ewidth = ewidth
    def __inject__(self):
        return "bv_sign_extend({}, {})".format(self.ewidth, self.var.__inject__())
    def __to__vlog__(self, prefix=""):
        return VSignExtend(self.var.__to__vlog__(prefix), self.ewidth)

class UclidFunctionApply(UclidExpr):
    def __init__(self, func, arglist):
        super().__init__()
        self.iname = func if isinstance(func, str) else func.name
        self.arglist = arglist
    def __inject__(self):
        return "{}({})".format(self.iname, ', '.join([arg.__inject__() for arg in self.arglist]))
    def __to__vlog__(self, prefix=""):
        return VFuncApplication(self.iname, [arg.__to__vlog__(prefix) for arg in self.arglist])

class UclidArraySelect(UclidExpr):
    def __init__(self, array, indexseq):
        super().__init__()
        self.iname = array if isinstance(array, str) else array.__inject__()
        self.indexseq = [ind if isinstance(ind, UclidExpr) else UclidLiteral(str(ind)) for ind in indexseq]
    def __inject__(self):
        return "{}[{}]".format(self.iname, "][".join([ind.__inject__() for ind in self.indexseq]))
    def __to__vlog__(self, prefix=""):
        return VArraySelect(prefix+self.iname, [arg.__to__vlog__(prefix) for arg in self.indexseq])

class UclidArrayUpdate(UclidExpr):
    def __init__(self, array, index, value):
        super().__init__()
        self.iname = array if isinstance(array, str) else array.__inject__()
        self.index = index if isinstance(index, UclidExpr) else UclidLiteral(str(index))
        self.value = value if isinstance(value, UclidExpr) else UclidLiteral(str(value))
    def __inject__(self):
        return "{}[{} -> {}]".format(self.iname, self.index.__inject__(), self.value.__inject__())

class UclidBVExtract(UclidExpr):
    def __init__(self, bv: UclidExpr, high, low):
        super().__init__()
        self.bv = bv
        self.high = high
        self.low = low
    def __inject__(self):
        return "({})[{}:{}]".format(self.bv.__inject__(), self.high, self.low)
    def __to__vlog__(self, prefix=""):
        return VSlice(self.bv.__to__vlog__(prefix), self.high, self.low)

class UclidRecordSelect(UclidExpr):
    def __init__(self, recvar: UclidExpr, elemname: str):
        super().__init__()
        self.recvar = recvar
        self.elemname = elemname
    def __inject__(self):
        return "{}.{}".format(self.recvar.__inject__(), self.elemname)

class UclidRecordUpdate(UclidExpr):
    def __init__(self, recvar: UclidExpr, elemname: str, value: UclidExpr):
        super().__init__()
        self.recvar = recvar
        self.elemname = elemname
        self.value = value
    def __inject__(self):
        return "{}[{} := {}]".format(self.recvar.__inject__(), self.elemname, self.value.__inject__())

class UclidForall(UclidExpr):
    def __init__(self, variable, typ, expr : UclidExpr):
        super().__init__()
        self.variable = variable
        self.typ = typ
        self.expr = expr
    def __inject__(self):
        return "forall ({} : {}) :: ({})".format(self.variable, self.typ.__inject__(), self.expr.__inject__())

class DeclTypes(Enum):
    VAR = 0
    FUNCTION = 1
    TYPE = 2
    INSTANCE = 3
    SYNTHFUN = 4
    DEFINE = 5
    CONST = 6

    # def __inject__(self):
    #     if self.value == 0:
    #         return "var"
    #     elif self.value == 1:
    #         return "function"
    #     elif self.value == 2:
    #         return "type"
    #     elif self.value == 3:
    #         return "instance"
    #     elif self.value == 4:
    #         return "synthesis function"
    #     elif self.value == 5:
    #         return "define"
    #     elif self.value == 6:
    #         return "const"
    #     else:
    #         logging.error("Unsupported decl type {}".format(self.name))
    #         return ""

# Base class for (all sorts of) uclid declarations
class UclidDecl(UclidElement):
    def __init__(self, decltype) -> None:
        super().__init__()
    # Declarations should additionally support the __decl__ function
        self.decltype = decltype
    def __inject__(self):
        if self.decltype == DeclTypes.VAR:
            return "{} {} : {};".format(self.porttype.name, self.name, self.__declstring__)
        elif self.decltype == DeclTypes.TYPE:
            return "type {} = {};".format(self.name, self.__declstring__)
        elif self.decltype == DeclTypes.INSTANCE:
            return "instance {} : {};".format(self.name, self.__declstring__)
        elif self.decltype == DeclTypes.CONST:
            return "const {} : {};".format(self.name, self.__declstring__)
        # TODO: add support for other types
        # TODO: add full type system
        else:
            logging.error(f"Declaration for decltype {self.decltype} is not permitted")
            exit(1)
    # Inject is declaration specific an needs to be overloaded
    @property
    def __declstring__(self):
        raise NotImplementedError

class UclidType(UclidElement):
    def __init__(self, typestring):
        self.typestring = typestring
    def __inject__(self):
        return self.typestring

class UclidBooleanType(UclidType):
    def __init__(self):
        super().__init__("boolean")
UBool = UclidBooleanType

class UclidIntegerType(UclidType):
    def __init__(self):
        super().__init__("integer")
UInt = UclidIntegerType

class UclidBVType(UclidType):
    def __init__(self, width):
        super().__init__("bv{}".format(width))
        self.width = width
UBV = UclidBVType

class UclidArrayType(UclidType):
    def __init__(self, itype, etype):
        super().__init__("[{}]{}".format(itype.__inject__(), etype.__inject__()))
        self.indextype = itype
        self.elemtype = etype

class UclidEnumType(UclidType):
    def __init__(self, members):
        super().__init__("enum {{ {} }}".format(", ".join(members)))
        self.members = members

class UclidSynonymType(UclidType):
    def __init__(self, name):
        super().__init__(name)
        self.name = name

class UclidUnintType(UclidType):
    def __init__(self, name):
        super().__init__(name)
        self.name = name

class UclidTypeDecl(UclidDecl):
    def __init__(self, name: str, typexp) -> None:
        super().__init__(DeclTypes.TYPE)
        self.name = name
        self.typexp = typexp
    def __declstring__(self):
        return self.typexp.__inject__()

class UclidFunctionSig(UclidElement):
    def __init__(self, ip_args, out_type) -> None:
        super().__init__()
        self.ip_args = ip_args
        self.out_type = out_type
    def __inject__(self):
        input_sig = ', '.join(["{} : {}".format(i[0], i[1].__inject__()) for i in self.ip_args])
        return "({}) : {}".format(input_sig, self.out_type.__inject__())


class UclidDefine(UclidElement):
    """
    Define (function) declarations
    """
    def __init__(self, name: str, function_sig: UclidFunctionSig, body : UclidExpr) -> None:
        super().__init__()    
        self.name = name
        self.function_sig = function_sig
        self.body = body
        UclidContext.__add_definedecl__(self)
    def __inject__(self):
        return "\tdefine {}{} = {};".format(
            self.name, self.function_sig.__inject__(), self.body.__inject__())     

# A named literal (WYSIWYG)
class UclidLiteral(UclidExpr):
    """
    Literals and variables in Uclid
    """
    def __init__(self, lit, isprime = False) -> None:
        super().__init__()
        self.lit = lit if isinstance(lit, str) else str(lit)
        self.isprime = isprime
    def p(self):
        if self.isprime:
            logging.warn("Double prime for literal {}".format(self.lit))
        return UclidLiteral(self.lit + '\'', True)
    def __inject__(self):
        return self.lit
    def __add__(self, other):
        return super().__add__(other)
    def __to__vlog__(self, prefix=""):
        return VLiteral(str(self.lit))

class UclidIntegerConst(UclidLiteral):
    def __init__(self, val):
        super().__init__(val)
class UclidBooleanConst(UclidLiteral):
    def __init__(self, val : bool):
        super().__init__(str(val).lower())
class UclidBVConst(UclidLiteral):
    def __init__(self, val, width: int):
        super().__init__(val)
        self.width = width
        self.lit = f'{self.lit}bv{str(self.width)}'
    def __to__vlog__(self, prefix=""):
        return VBVConst(self.val, self.width)

# Uclid Const declaration
class UclidConst(UclidLiteral):
    def __init__(self, name: str, typ : UclidType, value=None):
        super().__init__(name)
        self.name = name
        self.typ = typ
        if value is None:
            self.declstring = typ.name
        else:
            self.declstring = "{} = {}".format(typ.name, value.__inject__() if isinstance(value, UclidExpr) else str(value))
        _ = UclidDecl(self, DeclTypes.CONST)
    def __inject__(self):
        return self.name

# ==============================================================================
# Uclid Variables
# ==============================================================================
class PortType(Enum):
    var = 0
    input = 1
    output = 2

    def __inject__(self):
        if self.value == PortType.var:
            return "var"
        elif self.value == PortType.input:
            return "input"
        elif self.value == PortType.output:
            return "output"
        else:
            logging.error("Unsupported port type")

# Uclid Var declaration
class UclidVar(UclidLiteral):
    def __init__(self, name, typ : UclidType, porttype=PortType.var):
        super().__init__(name)
        self.name = name
        self.typ = typ
        self.porttype = porttype
        self.declstring = typ.name
        _ = UclidDecl(self, DeclTypes.VAR)
    def __inject__(self):
        return self.name
    def __add__(self, other):
        return super().__add__(other)
    def __to__vlog__(self, prefix=""):
        if isinstance(self.typ, UclidBVType):
            return VLiteral(prefix+self.name)
        elif isinstance(self.typ, UclidBooleanType):
            return VLiteral(prefix+self.name)
        elif isinstance(self.typ, UclidArrayType):
            if isinstance(self.typ.elemtype, UclidBVType) and isinstance(self.typ.indextype, UclidBVType):
                return VReg(prefix+self.name, self.typ.elemtype.width, pow(2, self.typ.indextype.width))
            else:
                logging.error("Verilog generation for non-compatible type: {}".format(self.typ.declstring))
        else:
            logging.error("Verilog generation for non-compatible type: {}".format(self.typ.declstring))
def mkVar(varname : list, typ, porttype):
    return UclidVar(varname, typ, porttype)
def mkVars(varnames : list, typ, porttype):
    return [UclidVar(varname, typ, porttype) for varname in varnames]

# Uclid integer type declaration
class UclidIntegerVar(UclidVar):
    def __init__(self, name, porttype=PortType.var) -> None:
        super().__init__(name, UclidIntegerType(), porttype)
    def __add__(self, other):
        return super().__add__(other)
def mkIntVar(varname : str, porttype=PortType.var):
    return UclidIntegerVar(varname, porttype)
def mkIntVars(varnames : list, porttype=PortType.var):
    return [UclidIntegerVar(varname, porttype) for varname in varnames]
# Uclid bitvector type declaration
class UclidBVVar(UclidVar):
    def __init__(self, name, width, porttype=PortType.var) -> None:
        super().__init__(name, UclidBVType(width), porttype)
        self.width = width
    def __add__(self, other):
        return super().__add__(other)

class UclidBooleanVar(UclidVar):
    def __init__(self, name, porttype=PortType.var) -> None:
        super().__init__(name, UclidBooleanType(), porttype)
    def __add__(self, _):
        logging.error("Addition not supported on Boolean type")
        # return super().__add__(other)

# ==============================================================================
# Uclid statements
# ==============================================================================
class UclidStmt(UclidElement):
    """
        Statements in Uclid.
    """
    def __init__(self):
        pass
    def __inject__(self):
        raise NotImplementedError
    def __to__vlog__(self, prefix=""):
        raise NotImplementedError

class UclidComment(UclidStmt):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text
    def __inject__(self):
        return "\t//{}\n".format(self.text)
class UclidRaw(UclidStmt):
    def __init__(self, stmt : str):
        super().__init__()
        self.stmt = stmt
    def __inject__(self):
        return self.stmt
class UclidEmpty(UclidRaw):
    def __init__(self):
        super().__init__("")
class UclidLocalVarInstStmt(UclidStmt):
    def __init__(self, name, typ):
        super().__init__()
        self.name = name
        self.typ = typ
    def __inject__(self):
        return "var {} : {};".format(self.name, self.typ.__inject__())
class UclidAssignStmt(UclidStmt):
    def __init__(self, lval, rval) -> None:
        super().__init__()
        self.lval = lval
        if isinstance(rval, UclidExpr): 
            self.rval = rval
        elif isinstance(rval, int):
            self.rval = UclidLiteral(str(rval))
        elif isinstance(rval, str):
            self.rval = UclidLiteral(rval)
        else:
            logging.error("Unsupported rval type {} in UclidAssign".format(
                type(rval)
            ))
    def __inject__(self):
        return "{} = {};".format(self.lval.__inject__(), self.rval.__inject__())
    def __to__vlog__(self, prefix=""):
        return VBAssignment(self.lval.__to__vlog__(prefix), self.rval.__to__vlog__(prefix))

class UclidBlockStmt(UclidStmt):
    def __init__(self, stmts = [], indent=0) -> None:
        super().__init__()
        self.stmts = stmts
        self.indent = indent
    def __inject__(self):
        return ("\n" + self.indent*"\t").join([stmt.__inject__() for stmt in self.stmts])
    def __to__vlog__(self, prefix=""):
        return VStmtSeq([s.__to__vlog__(prefix) for s in self.stmts])

class UclidCaseStmt(UclidStmt):
    def __init__(self, conditionlist, stmts):
        if not isinstance(conditionlist, list) or not isinstance(stmts, list):
            logging.error("UclidCase requires a pair of lists denoting case-split conditions and statements")
            print([cond.__inject__() for cond in conditionlist])
        self.conditionlist = conditionlist
        self.stmts = stmts
    def __inject__(self):
        cases = ['({})\t: {{ \n{} \n}}'.format(item[0].__inject__(), textwrap.indent(item[1].__inject__(), '\t'))
            for item in zip(self.conditionlist, self.stmts)]
        return textwrap.dedent(
                """
                case
                {}
                esac
                """
            ).format("\n".join(cases))

class UclidITEStmt(UclidStmt):
    def __init__(self, condition, tstmt, estmt=None):
        self.condition = condition
        self.tstmt = tstmt
        self.estmt = estmt

    def __inject__(self):
        if self.estmt == None:
            return '''
    if ({}) {{ {} }}
        '''.format(self.condition.__inject__(), self.tstmt.__inject__())
        else:
            return '''
    if ({}) {{ {} }} else {{ {} }}
        '''.format(self.condition.__inject__(), self.tstmt.__inject__(), self.estmt.__inject__())
    def __to__vlog__(self, prefix=""):
        return VITE(
            self.condition.__to__vlog__(prefix),
            self.tstmt.__to__vlog__(prefix),
            self.estmt.__to__vlog__(prefix),
        )

class UclidITENestedStmt(UclidStmt):
    def __init__(self, conditionlist, stmtlist):
        if len(conditionlist) == len(stmtlist):
            self.format = "IT"
        elif len(conditionlist) == len(stmtlist) - 1:
            self.format = "ITE"
        else:
            logging.error("Illegal lengths of conditionlist and stmt blocks in ITE operator")
        self.conditionlist = conditionlist
        self.stmtlist = stmtlist

    def __inject__(self):
        def ite_rec(clist, slist):
            if len(clist) > 0 and len(slist) > 0:
                nesting = ite_rec(clist[1:], slist[1:])
                return '''
    if ({}) {{ {} }} 
    else {{ {} }}'''.format(clist[0].__inject__(), slist[0].__inject__(), nesting)
            elif len(slist) > 0:
                return "{}".format(slist[0].__inject__())
            elif len(clist) == 0:
                return ""
            else:
                logging.error("Mismatched clist/slist lengths in ite_rec")
        return ite_rec(self.conditionlist, self.stmtlist)

class UclidProcedureCallStmt(UclidStmt):
    def __init__(self, proc, ip_args, ret_vals):
        super().__init__()
        self.iname = proc if isinstance(proc, str) else proc.name
        self.ip_args = ip_args
        self.ret_vals = ret_vals
    def __inject__(self):
        return "call ({}) = {}({});".format(
            ', '.join([ret.__inject__() for ret in self.ret_vals]),
            self.iname, ', '.join([arg.__inject__() for arg in self.ip_args]))
class UclidInstanceProcedureCallStmt(UclidStmt):
    def __init__(self, instance, proc, ip_args, ret_vals):
        super().__init__()
        self.instance = instance if isinstance(instance, str) else instance.name
        self.iname = '{}.{}'.format(self.instance, proc if isinstance(proc, str) else proc.name)
        self.ip_args = ip_args
        self.ret_vals = ret_vals
    def __inject__(self):
        return "call ({}) = {}({});".format(
            ', '.join([ret.__inject__() for ret in self.ret_vals]),
            self.iname, ', '.join([arg.__inject__() for arg in self.ip_args]))
class UclidNextStmt(UclidStmt):
    def __init__(self, instance):
        super().__init__()
        self.instance = instance

    def __inject__(self):
        return "next ({});".format(self.instance.__inject__())

class UclidAssumeStmt(UclidStmt):
    def __init__(self, body):
        super().__init__()
        self.body = body
    def __inject__(self):
        return "assume({});".format(self.body.__inject__())

class UclidAssertStmt(UclidStmt):
    def __init__(self, body):
        super().__init__()
        self.body = body
    def __inject__(self):
        return "assert({});".format(self.body.__inject__())

class UclidHavocStmt(UclidStmt):
    def __init__(self, variable):
        super().__init__()
        self.variable = variable
    def __inject__(self):
        return "havoc {};".format(self.variable.__inject__())

class UclidForStmt(UclidStmt):
    def __init__(self, range_variable, range_typ, range_low, range_high, body):
        super().__init__()
        self.range_high = range_high
        self.range_low  = range_low
        self.range_variable = range_variable
        self.range_typ = range_typ
        self.body = body
    def __inject__(self):
        return '''
for ({} : {}) in range({}, {}) {{
    {}
}}
'''.format(self.range_variable.__inject__(), self.range_typ.__inject__(), self.range_low.__inject__(), self.range_high.__inject__(), self.body.__inject__())


# ==============================================================================
# Uclid sub-module instances
# ==============================================================================
class UclidInstance(UclidElement):
    """
    Module instances in Uclid
    """
    def __init__(self, name, module, argmap) -> None:
        super().__init__()
        self.name = name
        self.module = module
        self.argmap = argmap
        modname = module.name
        # print(argmap)
        # print([i.__inject__() for i in UclidContext.ip_var_decls[modname]])
        self.declstring = "{}({})".format(modname, 
            ", ".join(["{} : ({})".format(port.name, argmap[port.name].__inject__()) for port in UclidContext.ip_var_decls[modname] + UclidContext.op_var_decls[modname]]))
        self.decl = UclidDecl(self, DeclTypes.INSTANCE)
    def __inject__(self):
        return self.name
class UclidInstanceRaw(UclidElement):
    """
        Raw (external) module instances in Uclid
    """
    def __init__(self, name, module, argmap) -> None:
        super().__init__()
        self.name = name
        self.argmap = argmap
        modname = module.name if isinstance(module, UclidModule) else module
        # print(argmap)
        # print([i.__inject__() for i in UclidContext.ip_var_decls[modname]])
        self.declstring = "{}({})".format(modname, 
            ", ".join(["{} : ({})".format(portname, argmap[portname]) for portname in argmap]))
        self.decl = UclidDecl(self, DeclTypes.INSTANCE)
    def __inject__(self):
        return self.name
class UclidInstanceVarAccess(UclidExpr):
    def __init__(self, instance, var):
        self.instance = instance.__inject__() if isinstance(instance, UclidModule) else instance
        self.var = var
    def __inject__(self):
        return "{}.{}".format(self.instance, self.var.__inject__())


# ==============================================================================
# Uclid Module Imports
# ==============================================================================
class UclidImportDecl(UclidDecl):
    def __init__(self, decltype, name, modulename, refname) -> None:
        super().__init__(decltype)
        self.name = name
        self.modulename = modulename
        self.refname = refname
    def __declstring__(self):
        return f"{self.modulename}.{self.refname}"
class UclidWildcardImportDecl(UclidImportDecl):
    def __init__(self, decltype, modulename) -> None:
        super().__init__(decltype, "*", modulename, "*")

# ==============================================================================
# Uclid Procedures
# ==============================================================================

class UclidProcedureSig(UclidElement):
    # ip_args are pairs of str, Uclidtype elements
    def __init__(self, ip_args, modify_vars, return_vars, requires, ensures) -> None:
        super().__init__("")
        self.ip_args = ip_args
        self.modify_vars = modify_vars
        self.return_vars = return_vars
        self.requires = requires
        self.ensures = ensures

    def __inject__(self):
        input_str = ', '.join(["{} : {}".format(i[0] if isinstance(i[0], str) else i[0].lit,  i[1].__inject__()) for i in self.ip_args])
        modify_str = "\nmodifies {};".format(
            ', '.join([sig.__inject__() for sig in self.modify_vars])
        ) if len(self.modify_vars) != 0 else ''
        return_str = "\nreturns ({})".format(', '.join(["{} : {}".format(i[0], i[1].__inject__()) for i in self.return_vars])) if len(self.return_vars) != 0 else ''
        ensures_str = "\nensures ({})".format(self.ensures.__inject__()) if self.ensures is not None else ''
        requires_str = "\nrequires ({})".format(self.requires.__inject__()) if self.requires is not None else ''
        return "({}){}{}{}{}".format(input_str, modify_str, requires_str, ensures_str, return_str)
class UclidProcedureDecl(UclidDecl):
    def __init__(self, name: str, typ: UclidProcedureSig, body: UclidStmt):
        super().__init__()
        self.name = name
        self.typ = typ
        self.body = body
    def __inject__(self):
        return """procedure {} 
    {}
{{
{} 
}}
    """.format(self.name, self.typ.__inject__(), textwrap.indent(self.body.__inject__(), '\t'))

# ==============================================================================
# Uclid Spec (assertions)
# ==============================================================================
class UclidSpecDecl(UclidDecl):
    def __init__(self, name, body, is_ltl=False) -> None:
        super().__init__()
        self.name = name
        self.body = body
        self.is_ltl = is_ltl
    def __inject__(self):
        if not self.is_ltl:
            return "property {} : {};\n".format(self.name, self.body.__inject__())
        return "property[LTL] {} : {};\n".format(self.name, self.body.__inject__())

# ==============================================================================
# Uclid Axiom (assumptions)
# ==============================================================================
class UclidAxiomDecl(UclidDecl):
    def __init__(self, body) -> None:
        super().__init__()
        self.body = body
    def __inject__(self):
        return "axiom ({});".format(self.body.__inject__())

class UclidControlCommand(UclidElement):
    """
    Uclid control block commands.
    """
    def __init__(self):
        pass
    def __inject__(self):
        raise NotImplementedError


class UclidControlBlock(UclidControlCommand):
    def __init__(self, stmts=[]):
        super().__init__()
        self.stmts = stmts
    def add(self, stmt):
        self.stmts.append(stmt)
    def __inject__(self):
        return '''
control {{
{}
}}'''.format(textwrap.indent("\n".join([stmt.__inject__() for stmt in self.stmts]), '\t'))

class UclidUnrollCommand(UclidControlCommand):
    def __init__(self, name, depth):
        self.depth = depth
        self.name = name
    def __inject__(self):
        return "{} = unroll({});".format(self.name, self.depth)

class UclidBMCCommand(UclidControlCommand):
    def __init__(self, name, depth):
        self.depth = depth
        self.name = name
    def __inject__(self):
        return "{} = bmc({});".format(self.name, self.depth)

class UclidCheckCommand(UclidControlCommand):
    def __init__(self):
        super().__init__()
    def __inject__(self):
        return "check;"

class UclidPrintCexCommand(UclidControlCommand):
    def __init__(self, engine, trace_items = []):
        super().__init__()
        self.engine = engine
        self.trace_items = trace_items
    def __inject__(self):
        return "{}.print_cex({});".format(self.engine.name,
        ', '.join([item.__inject__() if isinstance(item, UclidExpr) else str(item)
        for item in self.trace_items]))

class UclidPrintCexJSONCommand(UclidControlCommand):
    def __init__(self, engine, trace_items = []):
        super().__init__()
        self.engine = engine
        self.trace_items = trace_items
    def __inject__(self):
        return "{}.print_cex_json({});".format(self.engine.name,
        ', '.join([item.__inject__() if isinstance(item, UclidExpr) else str(item)
        for item in self.trace_items]))

class UclidPrintResultsCommand(UclidControlCommand):
    def __init__(self):
        super().__init__()
    def __inject__(self):
        return "print_results;"

# LTL Operator Macros
def X(expr):
    return UclidOpExpr("next", [expr])
def F(expr):
    return UclidOpExpr("eventually", [expr])
def G(expr):
    return UclidOpExpr("globally", [expr])

CMD_check = UclidCheckCommand()
CMD_print = UclidPrintResultsCommand()
