---
# try also 'default' to start simple
theme: seriph
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://source.unsplash.com/collection/94734566/1920x1080
# apply any windi css classes to the current slide
class: 'text-center'
# https://sli.dev/custom/highlighters.html
highlighter: shiki
# show line numbers in code blocks
lineNumbers: true
# some information about the slides, markdown enabled
info: |
  ## Tensorflow.js 
  Quick Tensorflow.js presentation

# persist drawings in exports and build
drawings:
  persist: false
---

# Tensorflow.js

A quick intro and feature overview



<!--
The last comment block of each slide will be treated as slide notes. It will be visible and editable in Presenter Mode along with the slide. [Read more in the docs](https://sli.dev/guide/syntax.html#notes)
-->

---

# What is Tensorflow.js?

TensorFlow.js is a library for machine learning in JavaScript. It provides an easy to use API for Tensors, Layers, Optimizers and Loss Functions. 
<br>
<br>

### Two main NPM packages: 

1. `@tensorflow/tfjs`
2. `@tensorflow/tfjs-node`
    - `@tensorflow/tfjs-node-gpu` Linux package with GPU support.
    
<br>
<br>
<br>

### Best of all, native **Typescript** support. 


<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---

# Computing Backends and Environments

Tensorflow.js works in the browser and Node.js. 

<br/>
<br/>
<div grid="~ cols-2 gap-2" m="t-4">
<div>
Browser:

- Mobile and desktop devices.
- Runs on almost any browser.
- WebGL support. 100x faster than the vanilla CPU backend.
</div>

<div>
Node.js:

- Binds directly to the TF API in written in C.
- GPU with CUDA support.
</div>
</div>
<br/>
<br/>
<br/>
<br/>
All backends provide the same API, just different performance benefits. 

---
class: px-20
---


# Compared to the Python package. 

The `Layers` API of TensorFlow.js is modeled after `Keras` so it’s very similar. This allows for easy code migration and context switching between the two. 


### Example

<div grid="~ cols-2 gap-2" m="t-4">

```python
# Python:
import keras
import numpy as np

# Build and compile model.
model = keras.Sequential()
model.add(keras.layers.Dense(units=1, input_shape=[1]))
model.compile(optimizer='sgd', loss='mean_squared_error')

# Generate some synthetic data for training.
xs = np.array([[1], [2], [3], [4]])
ys = np.array([[1], [3], [5], [7]])

# Train model with fit().
model.fit(xs, ys, epochs=1000)

# Run inference with predict().
print(model.predict(np.array([[5]])))
```

```ts 
// JavaScript:
import * as tf from '@tensorlowjs/tfjs';

// Build and compile model.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
const ys = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);

// Train model with fit().
await model.fit(xs, ys, {epochs: 1000});

// Run inference with predict().
model.predict(tf.tensor2d([[5]], [1, 1])).print();
```

</div>

---
layout: image-right
image: https://source.unsplash.com/collection/94734566/1920x1080
---

# Compared to the Python package (continued). 

Both APIs are very similar as you can see. 

Python functions have named arguments. 
```python {all|2}
# Python:
keras.layers.Dense(units=1, inputShape=[1])
```

We use configuration objects in Javascript to provide the same flexibility. 
```ts {all|2}
// JavaScript:
tf.layers.dense({units: 1, inputShape: [1]});
```

---

# In production

A common pattern is to train models using the Python library (it provides a more mature and robust API) and export them as `SavedModel` format. This format bundles the complete TensorFlow program, including trained parameters and computation.
<br/>
<br/>
<br/>
<div grid="~ cols-2 gap-2" m="t-4">

<div>

<br/>

- No conversion needed. Usually needed converting models used on the browser. 
- `SavedModel` can execute even if the model uses APIs that are not supported in TensorFlow.js yet.
- Performance benefits.

</div>
<div>

![Benchmarks](https://lh4.googleusercontent.com/aTAHknwotexVqj_5sENZIKpsh-EsP8AuDaBupZEjuTBMzAcPbkuLP-LHuhvPoGpEmSCPpMr9MXj2up6GHbo0BNwzTY779GMzZx5EeljBNfkjQzUO-i5IO1XKMTuGQqcCYekjHZ_3)

</div>
</div>
---
preload: true
---

# Example: Train and export model


```python
# Create and train a new model instance.
model = GRU_MANY(...)
model.fit(x, y, epochs=5)

# Save the entire model as a SavedModel.
!mkdir -p saved_model
model.save('saved_model/gru/000')
```
---

# Example: Load in TFJS-Node

```ts
import * as tf from '@tensorlowjs/tfjs';

// Load saved model
const model = await tf.node.loadSavedModel('saved_model/gru/000')

// Create input tensors
const courses = tf.tensor([['32a4ab7d-9910-4b69-92d2-e488ae96c37d']])
const tags = tf.tensor([['16c185a3-6633-401b-8aab-dd05f2d08f9b']])

// Make predictions
const preds = model.predict({ courses, tags })

preds.predictions.print()
// [
//     '3ac9746a-9eb7-4c8a-af23-6cb38a130ef4',
//     '208d545a-90ce-4580-93f3-e94bb30cbc49',
//     'd253500a-096d-4f11-a1e1-aad602b9b73e',
//     'e24f6967-143f-4699-b374-10331f1d6fba',
//     '7caddc5f-39ba-4ffb-86ef-c370e90be9bf',
//     '5e9223c2-59d9-430b-9853-09241161a6b7',
//     ...
// ]
```
---

# Other Benefits/Opportunities

- Offline in-browser inference 
- Allows non-python developers to use Tensorflow. 
- Use the JS/Node ecosystem. 


---
layout: center
class: text-center
---

# Thanks!

[Documentation](https://www.tensorflow.org/js/tutorials) · [GitHub](https://github.com/tensorflow/tfjs) · [Demos](https://www.tensorflow.org/js/demos)
