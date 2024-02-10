

import unittest

from pyclid import *

inttype = UclidIntegerType()
booltype = UclidBooleanType()

class Test(unittest.TestCase):
    def makeModule(self):
        return UclidModule("main")
    
    def testAdd(self):
        m = self.makeModule()
        a = m.mkVar("a", inttype)
        b = m.mkVar("b", inttype)
        c = m.mkVar("c", inttype)
        init = UclidInit([
            UclidAssignStmt(a, UclidLiteral(1)),
            UclidAssignStmt(b, UclidLiteral(2)),
            UclidAssignStmt(c, a + b)
        ])
        m.setInit(init)
        self.assertTrue('var c : integer;' in m.__inject__())
    
    def testUninterpretedType(self):
        m = self.makeModule()
        t = m.mkUninterpretedType("mytype")
        a = m.mkVar("a", inttype)
        b = m.mkVar("b", inttype)
        c = m.mkVar("c", t)
        init = UclidInit([
            UclidAssignStmt(a, UclidLiteral(1)),
            UclidAssignStmt(b, UclidLiteral(2)),
            UclidAssignStmt(c, a + b)
        ])
        m.setInit(init)
        self.assertTrue('var c : mytype;' in m.__inject__())

    def testArrayType(self):
        m = self.makeModule()
        t = m.mkArrayType(inttype, inttype)
        s = m.mkArrayType("myarr", inttype, inttype)
        a = m.mkVar("a", inttype)
        b = m.mkVar("b", inttype)
        c = m.mkVar("c", s)
        init = UclidInit([
            UclidAssignStmt(a, UclidLiteral(1)),
            UclidAssignStmt(b, UclidLiteral(2)),
            UclidAssignStmt(c, a + b)
        ])
        m.setInit(init)
        self.assertTrue('var c : myarr;' in m.__inject__())

    def testConst(self):
        m = self.makeModule()
        a = m.mkVar("a", inttype)
        b = m.mkVar("b", inttype)
        c = m.mkVar("c", inttype)
        d = m.mkConst("d", inttype, UclidLiteral(3))
        init = UclidInit([
            UclidAssignStmt(a, UclidLiteral(1)),
            UclidAssignStmt(b, UclidLiteral(2)),
            UclidAssignStmt(c, a + b),
            UclidAssignStmt(d, UclidLiteral(3))
        ])
        m.setInit(init)
        self.assertTrue('const d : integer = 3;' in m.__inject__())

    
if __name__ == '__main__':
    unittest.main()

