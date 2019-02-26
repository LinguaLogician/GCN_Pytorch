import networkx as nx
import matplotlib.pyplot as plt
G=nx.karate_club_graph()
nx.draw(G, with_labels = True)
plt.show()

file = open('karate.cites','w')
for i in G.edges():
    file.write(str(i[0])+' '+str(i[1])+'\n')
file.close()
file = open('karate.content','w')
for i in G.nodes():
    file.write(str(i))
    temp = [0 for j in range(34)]
    temp[i] = 1
    for j in temp:
	file.write(" "+str(j))
    file.write(' 1\n')
file.close()
