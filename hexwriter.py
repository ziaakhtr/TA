import sys, binascii
open(r'd:\Zia\alt-dashboard.py', 'ab').write(binascii.unhexlify(sys.argv[1]))
