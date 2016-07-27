//
// Created by Yan Hang on 3/1/16.
//

#include "segment-graph.h"

namespace segment_gb{

	universe *segment_graph(int num_vertices, std::vector<edge>& edges, float c) {
		// sort edges by weight
		const int num_edges = (int)edges.size();
		std::sort(edges.begin(), edges.end(), [](const edge& a, const edge& b){return a.w < b.w;});
		// make a disjoint-set forest
		universe *u = new universe(num_vertices);

		// init thresholds
		std::vector<float> threshold((size_t)num_vertices);
		for (int i = 0; i < num_vertices; i++) {
			threshold[i] = c;
		}

		// for each edge, in non-decreasing weight order...
		for (int i = 0; i < num_edges; i++) {
			edge *pedge = &edges[i];

			// components connected by this edge
			int a = u->find(pedge->a);
			int b = u->find(pedge->b);
			if (a != b) {
				if ((pedge->w <= threshold[a]) &&
				    (pedge->w <= threshold[b])) {
					u->join(a, b);
					a = u->find(a);
					threshold[a] = pedge->w + c / u->size(a);
				}
			}
		}
		return u;
	}
}

