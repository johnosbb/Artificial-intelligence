from sklearn.preprocessing import OneHotEncoder

x = [["Normal"], ["Error"]]
y = OneHotEncoder().fit_transform(x).toarray()
print(y)