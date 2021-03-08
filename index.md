# Bachelor thesis


## Welcome
This is my page for bachelor thesis

### Name
Machine learning for nonlocal games

### Supervisor
Mgr. Daniel Nagaj, PhD.

### Anotation
Nonlocal games are a key concept in quantum information, utilized from complexity theory to certification of quantum devices. They involve two or more players that win if they provide properly correlated answers to questions. The typical example is the CHSH game, related to Bell inequalities that can be violated by quantum mechanics. In this game, quantum players have a higher winning probability than classical players. Actually determining the optimal winning probability is a difficult problem in general. In this thesis, the student will investigate a variety of nonlocal games and search for optimal quantum strategies with the help of machine learning strategies (reinforcement learning).


### Goal
Optimalization of quantum strategies for non-local CHSH games via machine learning and evolutionary algorithms.

### Code

<a href="CHSH - code.zip">Download File</a>

### Bachelor thesis

<a href="Bachelor_Thesis.pdf">Download File</a>

### Structure of text - Main chapters


<ol>
<li>Introduction  (what is the problem? Why it is important? Why is is interesting – non locality? FTL information transmition?  Hook reader - storytelling method)
</li>
<li>Quantum Mechanics </li>
<li>Non local games ( problem in details)<ol>
      <li>CHSH</li>
      <li>Complexity</li>
    </ol></li>
<li>Reinforcement machine learning  (How are we going to solve this? Method ) <ol>
      <li>Markov Decision process</li>
      <li>Algortihm used</li>
      <li>Complexity</li>
    </ol></li>
<li>Optimalization (How are we going to solve this?  Simulated annealing / evolutionary algortihm)</li>
<li>Implementation (How have we solved this?)</li>
       <li>Analysis of obtained results<ol>
      <li>CHSH 2-player</li>
      <li>CHSH multiplayer</li>
              <li>Larger games</li>
              <li>Comparison quantum vs. classic</li>
    </ol></li>
 <li>Discussion</li>
 <li>Conclusion</li>
</ol>

### Diary (in Slovak)

<details>
<summary>Click and see!</summary>
<ul>
<li> 16.2 - 23.2 </li>
<ul>
<li> Implementoval som GPU tensorflow model do trénovania môjho Reinforcement Agenta.</li>
<li> Pracoval som na funkcii, ktorá porovnáva najlepšiu klasickú a najlepšiu kvantovú taktiku. A vyberie také hry, ktoré majú najväčšie rozdiely.</li>
<li> Pridal som nové actions, ktoré vie vykonávať agent. (spomaľ, zrýchli) </li>
<li> Refactoring a väčšia abstrakcia hrier, genetických algoritmov etc. </li>
</ul>
<li> 23.2 - 2.3 </li>
<ul>
<li> Implementoval som PyTorch Deep Reinforcement DQN Agenta</li>
<li> Pracoval som na funkcii, ktorá porovnáva najlepšiu klasickú a najlepšiu kvantovú taktiku. A vyberie také hry, ktoré majú najväčšie rozdiely - upravil som ju, aby fungovala spravne.</li>
<li> Stretol som sa so skolitelom - urcili sme si uz finalne ciele mojej bakalarky</li>
<li> Refactoring. </li>
<li> Pracoval som na kapitole NonLocal games v LaTeXu</li>
</ul>
<li> 2.3 - 9.3 </li>
<ul>
<li> Snazil som sa optimalizovat DQN agenta</li>
<li> Implementoval som databazu, do ktorej sa budu ukladat uz preskumane hry, a ak sa znovu preskumaju, tak upsert ak sa najde lepsia hodnota.</li>
<li> Stretol som sa so skolitelom - zhodnotili sme tohtotyzdnovu pracu</li>
<li> Refactoring v triede Environment. </li>
<li> Urobil som state diagram, ako sa uci reinforcement learning, ako je to strukturovane </li>
<li> Viacere parametre som povytiahol von, nech si to pouzivatel moze sam nastavit </li>
<li> Pracoval som na kapitole NonLocal games v LaTeXe</li>
<li> Pracoval som na kapitole Reinforcement learning v LaTeXe</li>
</ul>
</ul>
</details>

### Deadlines, Milestones
#### * web page (do 11:30 29.10.2020) -> Done on time

#### * set milestones and deadlines (do 11:30 10.11.2020)  -> Done on time

#### * collecting sources(do 11:30 01.12.2020) -> Done on time

#### * 10 pages and prototype (do 15.1.2021) -> Done on time
  
#### * final program (do 15.3.2021)

#### * completing main part of bachelor thesis (do 15.4.2021)

#### * final testing and submitting (do 15.5.2021)

<a href="https://www.canva.com/design/DAEPEqLIsWM/ij-WJ0Wpchf-UAXgLVFSWA/view?utm_content=DAEPEqLIsWM&utm_campaign=designshare&utm_medium=link&utm_source=sharebutton">Presentation of sources, and, plan and progress of work</a>

### Sources

1. Two-player entangled games are NP-hard, Anand Natarajan, Thomas Vidick, Proceedings of CCC'18, arXiv:1710.03062
2. The Complexity of Entangled Games, Thomas Vidick, PhD thesis, UC Berkeley 2011, https://digitalassets.lib.berkeley.edu/etd/ucb/text/Vidick_berkeley_0028E_11907.pdf
3. CHSH hra/Bellove nerovnosti - Quantum Computation and Quantum Information, Nielsen Chuang, Vidick, najmä kapitola 2.6
4. Reinforcement learning https://www.udemy.com/course/artificial-intelligence-reinforcement-learning-in-python/
5. Quantum Physics, Marco Masi https://www.udemy.com/course/quantum-physics/
6. Quantum Computing, Michael Nielsen https://quantum.country/
7. Libraries - Qiskit, Numpy, Keras/PyTorch
8. Marco Masi.Quantum physics: an overviewof a weird world: a primer on the conceptualfoundations of quantum physics. Independentlypublished, 2019.
9. Urbain J Le Verrier. Theorie du mouvement demercure. InAnnales de l’Observatoire de Paris,volume 5, 1859.
10. Albert Einstein.  The photoelectric effect.Ann.Phys, 17(132):4, 1905.
11. John S Bell.  On the einstein podolsky rosenparadox.Physics Physique Fizika,  1(3):195,1964.
12. Alain Aspect.  Proposed experiment to test thenonseparability of quantum mechanics.Physicalreview D, 14(8):1944, 1976.
