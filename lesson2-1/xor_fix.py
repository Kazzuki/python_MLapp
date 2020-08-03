# ライブラリのインポート
# from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 学習用のデータと結果の準備
learn_data = [[0,0], [1,0], [0,1], [1,1]]
learn_label = [0, 1, 1, 0]

# 使うアルゴリズム(KNeighborsClassifier)
clf = KNeighborsClassifier(n_neighbors=1)
# 学習用データと結果の学習
clf.fit(learn_data, learn_label)

# テストデータによる予測
test_data = [[0,0], [1,0], [0,1], [1,1]]
test_label = clf.predict(test_data)

# 予測結果の評価
print(test_data, "の予想結果:", test_label)
print("正解率　＝", accuracy_score([0, 1, 1, 0], test_label))