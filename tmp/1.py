import tushare as ts

cds = ['002475', '603025', '000158']
d = ts.get_realtime_quotes(cds)
p = d['price'].astype(dtype=float)
c = d['pre_close'].astype(dtype=float)
rise = (p - c) / c * 100.
print(rise)
