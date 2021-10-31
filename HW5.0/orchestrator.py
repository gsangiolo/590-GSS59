import importlib

mode = 'wiki'

cleaner = importlib.import_module('01-clean').Cleaner(max_texts=50, mode=mode)
trainer = importlib.import_module('02-train').Trainer(mode=mode)
evaluator = importlib.import_module('03-evaluate').Evaluator(mode=mode)

cleaner.gather_text()
cleaner.split_chunks()

names, texts = trainer.read_clean_books_from_dir()
X_train, X_val, y_train, y_val = trainer.split_train_val(names, texts)
model, history = trainer.build_model(X_train, y_train, X_val, y_val)
trainer.evaluate_model(model, history, X_train, y_train, X_val, y_val)

#evaluator.evaluate_model()
