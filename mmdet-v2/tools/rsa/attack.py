import rsa
from rsa.key import PublicKey, PrivateKey


def is_belong(fn, n):
    assert n / 2 - 1 <= fn <= (n ** 0.5 - 1) ** 2


pub_key, pri_key = rsa.newkeys(64)
fn = (pri_key.p - 1) * (pri_key.q - 1)
is_belong(fn, pri_key.n)
MSG = str(rsa.encrypt("".encode(), pub_key))


# n/2 -1 <= f(n) <= (sqrt(n) - 1)^2
def crack_rsa(n, e):
    if isinstance(n, str): n = int(n, 16)
    if isinstance(e, str): e = int(e, 16)
    pri_k = PrivateKey()


crack_rsa(pub_key.n, pub_key.e)
