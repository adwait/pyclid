

from pyclid import UclidContext, UclidBooleanType, UclidIntegerType, UclidIntegerVar, PortType

# Variable creation sugaring
def mkVar(varname, typ, porttype):
    UclidContext.curr_module.mkVar(varname, typ, porttype)    
def mkVars(varnames : list, typ, porttype):
    return [UclidContext.curr_module.mkVar(varname, typ, porttype) for varname in varnames]
def mkIntVar(varname : str, porttype=PortType.var):
    return UclidContext.curr_module.mkIntegerVar(varname, porttype)
def mkIntVars(varnames : list, porttype=PortType.var):
    return [UclidContext.curr_module.mkIntegerVar(varname, porttype) for varname in varnames]

UBool = UclidBooleanType
UInt = UclidIntegerType

