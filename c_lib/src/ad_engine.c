#include "../include/ad_engine.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// --- Global Tape ---
static t_node **TAPE = NULL;
static size_t TAPE_SIZE = 0;
static size_t TAPE_CAPACITY = 0;

// Helper to add values to Tape
static void _add_to_graph(t_node *n)
{
	if (TAPE_SIZE >= TAPE_CAPACITY)
	{
		if (TAPE_CAPACITY == 0)
		{
			TAPE_CAPACITY = 128;
		} else {
			TAPE_CAPACITY = TAPE_CAPACITY * 2;
		}
		TAPE = (t_node**)realloc(TAPE, TAPE_CAPACITY * sizeof(t_node*));
		if (!TAPE)
		{
			perror("Failed to realloc tape");
			exit(EXIT_FAILURE);
		}
	}
	TAPE[TAPE_SIZE++] = n;
}

// Helper to create node
static t_node *_create_node(
		float data,
		t_node **parents,
		int num_parents,
		void (*backward_op)(t_node*)
		)
{
	t_node *n = (t_node*)malloc(sizeof(t_node));
	if (!n)
	{
		perror("Failed to allocate Node");
		exit(EXIT_FAILURE);
	}
	n->data = data,
	n->grad = 0.0;
	n->num_parents = num_parents;
	n->parents = NULL;
	if (num_parents > 0)
	{
		n->parents = (t_node**)malloc(num_parents * sizeof(t_node*));
		if (!n->parents)
		{
			perror("Failed to allocate parents array");
			free(n);
			exit(EXIT_FAILURE);
		}
		int i = 0;
		while (i < num_parents)
		{
			n->parents[i] = parents[i];
			i ++;
		}
	}
	n->_backward_op = backward_op;

	_add_to_graph(n);
	return n;
}

// --- Tape Management ---
void ad_destroy_tape()
{
	if (TAPE)
	{
		int i = 0;
		while (i < TAPE_SIZE)
		{
			if (TAPE[i])
			{
				free(TAPE[i]->parents);
				free(TAPE[i]);
			}
			i++;
		}
		free(TAPE);
		TAPE = NULL;
	}
	TAPE_SIZE = 0;
	TAPE_CAPACITY = 0;
}

void ad_init_tape(size_t initial_capacity)
{
	if (TAPE)
	{
		ad_destroy_tape();
	}
	if (initial_capacity > 0)
	{
		TAPE_CAPACITY = initial_capacity;
	} else {
		TAPE_CAPACITY = 128;
	}
	TAPE = (t_node**)malloc(TAPE_CAPACITY * sizeof(t_node*));
	if (!TAPE)
	{
		perror("Faled to allocate tape");
		exit(EXIT_FAILURE);
	}
	TAPE_SIZE = 0;
}

// --- Variable Creation ---
t_node *ad_create_variable(float value)
{
	return _create_node(value, NULL, 0, NULL); // data, parents, num_parents, backward_op
}

// --- Backward Operations ---
static void _backward_add(t_node *self)
{
	if (self->num_parents != 2) return;
	self->parents[0]->grad += self->grad;
	self->parents[1]->grad += self->grad;
}

static void _backward_mul(t_node *self)
{
	if (self->num_parents != 2) return;
	t_node *a = self->parents[0];
	t_node *b = self->parents[1];
	a->grad += self->grad * b->data;
	b->grad += self->grad * a->data;
}

// --- Forward Operations ---
t_node *ad_add(t_node *a, t_node *b)
{
	t_node *parents[] = {a, b};
	return _create_node(a->data + b->data, parents, 2, _backward_add);
}

t_node *ad_mul(t_node *a, t_node *b)
{
	t_node *parents[] = {a, b};
	return _create_node(a->data * b->data, parents, 2, _backward_mul);
}

// --- Backward Pass ---
void ad_backward(t_node *output_node, float seed_gradient)
{
	if (!output_node) return;
	output_node->grad = seed_gradient;
	int i = TAPE_SIZE;
	while (i >= 0)
	{
		t_node *n = TAPE[i];
		if (n && n->_backward_op)
		{
			n->_backward_op(n);
		}
		--i;
	}
}

// --- Gradient Retrieval ---
float ad_get_gradient(t_node *n)
{
	if (!n) return 0.0;
	return n->grad;
}

float ad_get_data(t_node *n)
{
	if (!n) return 0.0;
	return n->data;
}
