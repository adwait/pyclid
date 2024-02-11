

import unittest

from pyclid import *

inttype = UclidIntegerType()
booltype = UclidBooleanType()

class Test(unittest.TestCase):

    def testAdd(self):
        m = UclidModule("main")
        a = m.mkVar("a", inttype)
        b = m.mkVar("b", inttype)
        c = m.mkVar("c", inttype)
        d = m.mkVar("d", inttype, PortType.output)
        e = m.mkVar("e", inttype, PortType.output)
        init = UclidInitBlock([
            UclidAssignStmt(a, UclidLiteral(1)),
            UclidAssignStmt(b, UclidLiteral(2)),
            UclidAssignStmt(c, a + b)
        ])
        m.setInit(init)
        self.assertTrue('var c : integer;' in m.__inject__())
    
    def testUninterpretedType(self):
        m = UclidModule("main")
        t = m.mkUninterpretedType("mytype")
        a = m.mkVar("a", inttype)
        b = m.mkVar("b", inttype)
        c = m.mkVar("c", t)
        init = UclidInitBlock([
            UclidAssignStmt(a, UclidLiteral(1)),
            UclidAssignStmt(b, UclidLiteral(2)),
            UclidAssignStmt(c, a + b)
        ])
        m.setInit(init)
        self.assertTrue('var c : mytype;' in m.__inject__())

    def testArrayType(self):
        m = UclidModule("main")
        t = m.mkArrayType(inttype, inttype)
        s = m.mkArrayType("myarr", inttype, inttype)
        a = m.mkVar("a", inttype)
        b = m.mkVar("b", inttype)
        c = m.mkVar("c", s)
        init = UclidInitBlock([
            UclidAssignStmt(a, UclidLiteral(1)),
            UclidAssignStmt(b, UclidLiteral(2)),
            UclidAssignStmt(c, a + b)
        ])
        m.setInit(init)
        self.assertTrue('var c : myarr;' in m.__inject__())

    def testConst(self):
        m = UclidModule("main")
        a = m.mkVar("a", inttype)
        b = m.mkVar("b", inttype)
        c = m.mkVar("c", inttype)
        d = m.mkConst("d", inttype, UclidLiteral(3))
        init = UclidInitBlock([
            UclidAssignStmt(a, UclidLiteral(1)),
            UclidAssignStmt(b, UclidLiteral(2)),
            UclidAssignStmt(c, a + b),
            UclidAssignStmt(d, UclidLiteral(3))
        ])
        m.setInit(init)
        self.assertTrue('const d : integer = 3;' in m.__inject__())

    def testBig(self):
        m1 = UclidModule("subm")
        m1.mkVar("a", inttype)
        
        t1 = m1.mkUninterpretedType("mytype")
        t2 = m1.mkArrayType("arr_t", inttype, inttype)
        v1 = m1.mkVar("v1", t1)
        v2 = m1.mkVar("v2", t2)
        init1 = UclidInitBlock([
            UclidAssignStmt(UclidArraySelect(v2, [UclidLiteral(1)]), UclidLiteral(1)),
            UclidHavocStmt(v1)
        ])
        m1.setInit(init1)

        m2 = UclidModule("main")
        a = m2.mkVar("a", inttype)
        b = m2.mkVar("b", inttype)

        m2.mkImport(DeclTypes.TYPE, "t1", m1, "mytype")
        m2.setInit(UclidInitBlock([
            UclidAssignStmt(a, UclidLiteral(1)),
            UclidAssignStmt(b, UclidLiteral(2)),
            UclidAssumeStmt(UclidOpExpr("eq", [a, b]))
        ]))
        m2.setNext(UclidNextBlock([
            UclidAssignStmt(a.p(), UclidOpExpr("add", [a, b])),
            UclidAssignStmt(b.p(), UclidOpExpr("add", [a, UclidLiteral("1")])),
        ]))
        m2.setControl(UclidControlBlock([
            UclidBMCCommand("v", 10),
            UclidPrintCexJSONCommand("v"),
            UclidPrintResultsCommand()
        ]))

        print(m1.__inject__())
        print(m2.__inject__())

    
if __name__ == '__main__':
    unittest.main()

