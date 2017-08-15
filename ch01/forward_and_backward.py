import numpy as np

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
        return

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    # doutは逆伝播時の微分値
    # このメソッドは出力として微分値を返します
    def backword(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        # xとyは逆にします。
        # 逆にする理由は、x*yされた値を打ち消すための微分を作るため。
        # x=200,y=1.1というものに対してのbackwordでは、
        # (220になる)
        # xに対して1.1という値の打ち消し微分を渡したいので。
        return dx, dy

origin_price = 100
buy_count = 2
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
counted_price = mul_apple_layer.forward(x = origin_price, y = buy_count)
print("x = origin_price, y = buy_count at forward")
total_price = mul_tax_layer.forward(x= counted_price, y = tax)
print(origin_price, buy_count)
print("x = counted_price, y = tax")
print(counted_price, tax)
print("total_price")
print(total_price)

# backword
dprice = 1
dnon_tax_price, dtax = mul_tax_layer.backword(dout = dprice)
print("x = dnon_tax_price, y = dtax at backword")
print(dnon_tax_price, dtax)
dapple, dapple_num = mul_apple_layer.backword(dout = dnon_tax_price)
print(dapple, dapple_num, dtax)
