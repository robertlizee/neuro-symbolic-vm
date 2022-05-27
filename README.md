# neuro-symbolic-vm

This is the code accompaning the research paper http://arxiv.org/abs/2205.13440 The Neuro-Symbolic Brain.

## Notebooks

### Counter <a href="https://colab.research.google.com/github/robertlizee/neuro-symbolic-vm/blob/main/colab-notebooks/Counter.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

1. Creates 25000 prime attractors. 
2. Binds them in sequence in one shot.
3. Enumerates the prime attractors following the connection just learned.

### Hashtable <a href="https://colab.research.google.com/github/robertlizee/neuro-symbolic-vm/blob/main/colab-notebooks/Hashtable.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

1. Creates 5000 prime attractors.
2. Set the default for the neural hash table to 0.
3. Binds in one shot using the neural hash table the function (i * j) mod 5000 for i and j between 1 and 69.
4. Test to see if there is any error.

### TestVM <a href="https://colab.research.google.com/github/robertlizee/neuro-symbolic-vm/blob/main/colab-notebooks/TestVM.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

1. Train a neural VM with a test program.
2. Test the various programs: echo, addition, counting.
