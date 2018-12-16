

<h1>[Соревнование Telecom Data Cup][1]</h1>

* <h3> Используемые библиотеки:</h3>
    <a href = "https://scikit-learn.org/0.16/index.html">sklearn  </a>
    <p><a href = "https://docs.scipy.org/doc/numpy/user/index.html">numpy and skipy  </a>
    <p><a href = "http://pandas.pydata.org/pandas-docs/stable/">pandas  </a>
    <p><a href = "https://matplotlib.org/contents.html">matplotlib  </a>
    <p><a href = "http://seaborn.pydata.org/">seaborn  </a>
    <p><a href = "https://xgboost.readthedocs.io/en/latest/">xgboost  </a>
    <p><a href = "https://docs.azuredatabricks.net/applications/deep-learning/keras.html">keras  </a>
    <p><a href = "https://docs.python.org/2/library/pickle.html">pickle  </a>
    
    
    
* <h3> Установка и импортирование библиотек:</h3>
       1. Открываем командную строку и прописываем "pip install <название библиотеки>"
       <p>2. В коде прописываем import <название библиотеки>
  
* <h3> Описание кода:</h3>
       Удаляем повторяющиеся sk_id(номер абонента) для того чтобы каждой оценке соответствовал один номер абонента
       
       subs_features_train = subs_features_train.drop_duplicates('SK_ID') #удаляем дубликаты SK_ID
       subs_features_train.sort_values('SK_ID',ascending = True) # Сортируем таблицу по возрастанию SK_ID
       print(subs_features_train.shape)
  
   Создаем классификатор случайного леса, подбираем наиболее качественные параметры(глубину деревьев,количество функций для разделения и количество деревьев в лесу) и обучаем модель
       
       forest = RandomForestClassifier(random_state = 0)

        rfc_params = {
            'max_depth': np.arange(10, 30,10),
            'max_features': np.linspace(0.1, 5, 50),
            'n_estimators': list(np.arange(50,500,100))
        }
        rfc_grid = GridSearchCV(forest, rfc_params, cv=5, n_jobs=-1,error_score = "roc_auc")
        rfc_grid.fit(X_train, Y_train)
        
    Получаем признаки модели и точность по метрике ROC_AUC и accuracy. Делаем вывод , что случайный лес неэффективен для наших данных
    
    
    Создаем нейронную сеть с 4 слоями с разным количеством нейронов и разными функциями активации.После этого запускаем валидацию с количеством итераций = 6,количество эпох нейросети = 30. 
    
        def model_creator():
        model = models.Sequential()
        model.add(layers.Dense(120, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(layers.Dense(60, activation='relu'))
        model.add(layers.Dense(30, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics= ["acc"])
        return model

      estimators = []
      estimators.append(('standardize', StandardScaler()))
      estimators.append(('mlp', KerasClassifier(build_fn=model_creator, epoc hs=30, batch_size=3, verbose=1)))
      pipeline = Pipeline(estimators)
      results = cross_val_score(pipeline, X, y, cv=6, scoring='roc_auc')
      
      
    Последняя модель, которую мы рассмотрим - xgboost. После отбора параметров и обучения она показывает наилучшие результаты по метрике ROC_AUC.
      
      bst = xgb.train(param, dtrain, num_round)

      prob = bst.predict(deval)
      pred_train = pd.DataFrame(np.asarray([np.argmax(line) for line in prob]))
      print('Set Accuracy:', accuracy_score(Y_eval, pred_train))
      

      print('ROC_AUC:', roc_auc_score(Y_eval, pred_train))
   
   
[1]: https://mlbootcamp.ru/round/15/tasks/#19
