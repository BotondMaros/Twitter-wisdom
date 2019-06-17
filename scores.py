import math

prediction_numbers = [151,18,87,52,4,14,10,211,84,52]

VP = [80.6,100,83.3,50,100,66.7,71.4,66,60.9,95.8]
VR = [80.6,100,71.4,100,100,85.7,100,88,100,100]
VF1 = [80.6,100,76.8,66.6,100,75,83.3,75.4,75.6,97.8]

SP = [52.9,85.7,38.5,50,50,58.3,62.5,58.9,63.3,65.5]
SR = [87.1,100,71.4,100,100,100,100,82.5,73.8,91.3]
SF1 = [63.5,92.2,50,66.6,66.6,73.6,76.9,68.7,68.1,76.3]

FP = [54.7,85.7,50,50,50,58.3,71.4,63.4,69,56.1]
FR = [93.5,100,57.1,100,100,100,100,78.7,69,100]
FF1 = [69,92.2,53.3,66.6,66.6,73.6,83.3,70.2,69,71.8]

total_preds = sum(prediction_numbers)
rates = [prediction_numbers[i]/total_preds for i in range(len(prediction_numbers))]
def weighted_average(scores):
    return(sum(rates[g] * scores[g] for g in range(len(rates))))

print(weighted_average(VP))
print(weighted_average(VR))
print(weighted_average(VF1))
print(weighted_average(SP))
print(weighted_average(SR))
print(weighted_average(SF1))
print(weighted_average(FP))
print(weighted_average(FR))
print(weighted_average(FF1))