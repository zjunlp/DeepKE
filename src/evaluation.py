# coding=utf-8
classnum = 2


def load_data():
	preds = [1, 2, 0, 6]
	labels = [1, 2, 0, 0]
	return preds, labels


def evaluate_micro(preds, labels):
	pred_count = [0] * classnum
	label_count = [0] * classnum
	true_count = [0] * classnum
	for p, l in zip(preds, labels):
		l = int(l)
		label_count[l] += 1
		pred_count[p] += 1
		if p == l:
			true_count[l] += 1

	pred_count_sum = sum(pred_count)
	if pred_count_sum == 0:
		pred_count_sum = 1
	precision = sum(true_count) / float(pred_count_sum)
	recall = sum(true_count) / float(sum(label_count))
	if precision == 0 and recall == 0:
		f1 = 0.0
	else:
		f1 = 2 * precision * recall / float(precision + recall)
	return f1


def evaluate_micro_p_r_f1(preds, labels):
	pred_count = 0
	label_count = 0
	true_count = 0
	for p, l in zip(preds, labels):
		if (p != 0):
			pred_count += 1
		if (l != 0):
			label_count += 1
		if (p == l and p != 0):
			true_count += 1

	if pred_count == 0:
		pred_count = 1
	precision = true_count / float(pred_count)
	recall = true_count / float(label_count)
	if precision == 0 and recall == 0:
		f1 = 0.0
	else:
		f1 = 2 * precision * recall / float(precision + recall)
	# print("Micro-average F1: %0.02f%%"%(f1*100))
	return precision, recall, f1


def evaluate_macro(preds, labels):
	pred_count = [0] * classnum
	label_count = [0] * classnum
	true_count = [0] * classnum
	for p, l in zip(preds, labels):
		l = int(l)
		label_count[l] += 1
		pred_count[p] += 1
		if (p == l):
			true_count[l] += 1
	cnn = 0.0
	class_count = 0
	for t, l, p in zip(true_count, label_count, pred_count):
		if p == 0:
			p = 1
		precision = float(t) / p
		if l == 0:
			l = 1
		recall = float(t) / l
		if precision == 0 and recall == 0:
			f1 = 0
		else:
			f1 = (2 * precision * recall) / float(precision + recall)
		# print('t = %d\tp = %d\tl = %d\t'%(t,p,l))
		print("class %d : precision : %0.02f recall : %0.02f  f1 : %0.02f" % (
			class_count, precision * 100, recall * 100, f1 * 100))
		cnn += f1
		class_count += 1
	macro_f1 = cnn / float(classnum)
	# print("MACRO-average F1: %0.02f%%"%(macro_f1*100))
	return macro_f1


def evaluate(preds, labels):
	e1 = evaluate_micro(preds, labels)
	e2 = evaluate_macro(preds, labels)
	return e1, e2


if __name__ == '__main__':
	preds, labels = load_data()
	print(evaluate_micro(preds, labels))
	print(evaluate_macro(preds, labels))