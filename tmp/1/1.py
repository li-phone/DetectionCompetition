from gmpy2 import next_prime
from random import getrandbits
from libnum import n2s,s2n
# from secret import flag
flag='1233425235346346'
assert len(flag) <= 36
m, p, q = next_prime(s2n(flag)),next_prime(getrandbits(0x120)),next_prime(getrandbits(0x200))
s = m * p % q

print(hex(p))
print(hex(q))
print(hex(s))

# 0x49ee5ab58220093a5d3a3602533446e6ec03d7cc5f1b99dfa5a7a1a423cdb134f84a46d1
# 0x11115dc6324fc3edade1d84c6bd4cff6561550cecdfeca35815d02d3340281ab53f7a33bbe77949f0b6b98b35ddef346b52b359fa719b4b51eff0a52f92b7c5
# 0xd077b2bd82d807a07685e0777adf8745da78c5474af674ecde5a8094ada1385ce0ce2390763dc37351e6d6c92e520a6a1c34ab1cb1def0454d521ef822f9d9
