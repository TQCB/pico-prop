#ifndef AD_ENGINE_H
#define AD_ENGINE_H

#include <stddef.h>

// --- Value Struct --
typedef struct Node t_node;

struct Node
{
	float data;
	float grad;

	// Graph structure and backward pass
	t_node **parents; // array of points to parent nodes
	int num_parents;
	void (*_backward_op)(t_node *self); // function pointer to backward op
};

// --- Tape Management ---
void ad_init_tape(size_t initial_capacity); // initialize/reset tape to certain size
void ad_destroy_tape(); // free all values on tape
			
// --- Node Creation ---
t_node *ad_create_variable(float value);

// --- Operations ---
t_node *ad_add(t_node *a, t_node *b);
t_node *ad_sub(t_node *a, t_node *b);
t_node *ad_mul(t_node *a, t_node *b);
t_node *ad_div(t_node *a, t_node *b);

// --- Backwar Pass ---
void ad_backward(t_node *output_node, float seed_gradient); // seed is usually 1

// --- Gradient Retrieval ---
float ad_get_gradient(t_node *n);
float ad_get_data(t_node *n); // get forward pass result

#endif // AD_ENGINE_H
