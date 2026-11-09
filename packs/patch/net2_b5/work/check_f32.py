"""float32 forward pass (as a float32 runtime would do) vs float64 on recorded + synthetic states."""
import sys, os, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    sys.argv = ['x', 'work/r4.npz']; import verify2
z = np.load('work/r4.npz'); w32 = {k: z[k] for k in z.files}
w64 = tools.load('work/r4.npz')
for label, X in [('recorded', tools.states()['obs']), ('synthetic', verify2.synth(20000, seed=1).astype(np.float32))]:
    X32 = np.asarray(X, np.float32)
    h = np.tanh(X32 @ w32['W1'].T + w32['b1']); h = np.tanh(h @ w32['W2'].T + w32['b2']); lg32 = h @ w32['W3'].T + w32['b3']
    lg64 = tools.forward(w64, X32.astype(np.float64))
    print(label, 'argmax agreement f32 vs f64: %.5f' % (lg32.argmax(1) == lg64.argmax(1)).mean(), ' max |logit diff| %.2e' % np.abs(lg32 - lg64).max())
