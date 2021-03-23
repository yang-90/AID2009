# import re
# data = input()
# sp_data = re.split('[^a-zA-Z]+', data)
# print(sp_data)
# print(" ".join(sp_data[::-1]).strip())



class A(object):
    pass

class B(object):
    pass

class C(A):
    pass

a = A()
b = B()
c = C()

print(issubclass(B,A))
print(issubclass(A,A))
print(issubclass(C,(A,B)))
print(issubclass(C,object))
print("*"*30)

print(isinstance(B,A))
print(isinstance(a,A))
print(isinstance(c,A))
print(isinstance(C(),B))
print(isinstance(C(),(A,B)))

print("*"*30)



# func = lambda x : x%2
# result = filter(func,[1,2,3,4,5])
# print(list(result))

num = input()
num = reversed(num)
for i in num:
    print(i,end="")



#
# # -*- coding:utf-8 -*-
# class Solution:
#     def ReverseSentence(self, s):
#         # write code here
#         new_list = s.split(" ")
#         print(new_list)
#         res = []
#         for i in new_list:
#             res.append(i)
#         print(res)
#         result = res[::-1]
#         print(result)
#         return " ".join(result)
#
# s=Solution()
# print(s.ReverseSentence("nowcoder. a am I"))
# print(s.ReverseSentence("I am a nowcoder."))