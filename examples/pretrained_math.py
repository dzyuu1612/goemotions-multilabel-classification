"""Reproduce Appendix C.5/C.6 toy arithmetic; no model/data download.

This two-label SGD example explains the chain rule, not a trained Transformer
forward pass or the exact AdamW optimizer used for GoEmotions.
"""
import json
import numpy as np


def calculate():
    q=np.array([1.,0.]);k=np.eye(2);v=np.eye(2)
    scores=q@k.T/np.sqrt(2)
    attention=np.exp(scores-scores.max());attention/=attention.sum()
    h=np.array([.6,-.2]);w=np.array([1.2,-.4]);bias=.1
    logits=np.array([w@h+bias,-.4]);truth=np.array([1.,0.])
    probabilities=1/(1+np.exp(-logits))
    # Stable BCE, same mean reduction over the toy's two decisions.
    loss=float(np.mean(np.logaddexp(0,logits)-truth*logits))
    grad_z=(probabilities-truth)/truth.size
    grad_w=grad_z[0]*h
    eta=2e-5
    # Check the analytic chain rule against centered finite differences.
    eps=1e-5
    def objective(weight):
        z=np.array([weight@h+bias,-.4])
        return np.mean(np.logaddexp(0,z)-truth*z)
    numerical=np.array([(objective(w+eps*np.eye(2)[i])-objective(w-eps*np.eye(2)[i]))/(2*eps) for i in range(2)])
    np.testing.assert_allclose(grad_w,numerical,atol=1e-9,rtol=1e-6)
    return {'scope':'two-label toy, not benchmark or pretrained inference',
        'attention_scores':scores.tolist(),'attention_weights':attention.tolist(),
        'attention_output':(attention@v).tolist(),'logits':logits.tolist(),
        'probabilities':probabilities.tolist(),'mean_bce':loss,'gradient_logits':grad_z.tolist(),
        'gradient_weight_gratitude':grad_w.tolist(),'sgd_weight_new':(w-eta*grad_w).tolist(),
        'sgd_bias_new':float(bias-eta*grad_z[0]),'finite_difference_check_passed':True,
        'micro_batches_per_epoch':int(np.ceil(43410/16))}


def main():print(json.dumps(calculate(),indent=2))


if __name__=='__main__':main()
