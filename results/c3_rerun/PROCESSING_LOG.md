# Processing log

All three full training runs completed without restart. During postprocessing, the first full verification stopped while printing Vietnamese to the Windows cp1252 console. Metric computation had not failed. The console was set to UTF-8 and postprocessing resumed from the existing checked checkpoints and scores. No seed was retrained and no hyperparameter was changed. The failed processing log is retained in the delivery verification directory.
