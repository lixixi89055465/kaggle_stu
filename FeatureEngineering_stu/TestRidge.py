from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston

# 加载数据
bosten = load_boston()
features = bosten.data
target = bosten.target
print(features.shape)
print(target.shape)
# 特征标准化
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# 创建线形回归对象
ridge = Ridge(alpha=0.5)
# 拟合模型
model = ridge.fit(features_standardized, target)

print('1' * 100)
# 回归系数
print(model.coef_)
print("#" * 100)
# 多个超参数的设定
from sklearn.linear_model import RidgeCV

ridge_cv = RidgeCV(alphas=[0.1, 1.0, 10.0])
# 拟合模型
model = ridge_cv.fit(features_standardized, target)
print('2' * 100)
print(model.score(features_standardized, target))
print("2" * 100)
print(model.best_score_)
print("3" * 100)
print(model.alpha_)

print('#' * 100)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

