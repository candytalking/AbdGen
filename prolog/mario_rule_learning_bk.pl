% This file is a copyrighted under the BSD 3-clause licence, details of which can be found in the root directory.
% Code for Generating by Understanding: Neural Visual Generation with Logical Symbol Groundings
% https://arxiv.org/abs/2310.17451

:- use_module('meta_abd').
% :- style_check(-singleton).

%% grid size
width(2).
height(2).
my_goal([2,2]).
my_goal([1,2]).
my_goal([0,2]).


%% max program size
meta_abd:min_clauses(2).
meta_abd:max_clauses(4).

%% metarules
metarule(ident, [P,Q], [P,A,B], [[Q,A,B]]).
% metarule(chain, [P,Q,R], [P,A,B], [[Q,A,C],[R,C,B]]).
metarule(tailrec, [P,Q], [P,A,B], [[Q,A,C],[P,C,B]]).

%% tell metagol to use the BK
body_pred(up/2).
body_pred(down/2).
body_pred(left/2).
body_pred(right/2).
body_pred(terminate/2).

:- dynamic
    up/2,
    down/2,
    left/2,
    right/2,
    terminate/2.


%% abducibles
abducible(up/2).
abducible(down/2).
abducible(left/2).
abducible(right/2).
abducible(terminate/2).

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 Abduction
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
abduce(Atom, Old_abd_facts, New_abd_facts, Old_score, New_score) :-
    abduce_atom(Atom, Old_abd_facts, New_abd_facts, Old_score, New_score).

abduce_atom(Atom, Old_abd_facts, New_abd_facts, Old_score, New_score) :-
    Atom =.. [up, X, Y],
    abduce_up(X, Y, Old_abd_facts, New_abd_facts, Old_score, New_score).

abduce_atom(Atom, Old_abd_facts, New_abd_facts, Old_score, New_score) :-
    Atom =.. [down, X, Y],
    abduce_down(X, Y, Old_abd_facts, New_abd_facts, Old_score, New_score).

abduce_atom(Atom, Old_abd_facts, New_abd_facts, Old_score, New_score) :-
    Atom =.. [left, X, Y],
    abduce_left(X, Y, Old_abd_facts, New_abd_facts, Old_score, New_score).

abduce_atom(Atom, Old_abd_facts, New_abd_facts, Old_score, New_score) :-
    Atom =.. [right, X, Y],
    abduce_right(X, Y, Old_abd_facts, New_abd_facts, Old_score, New_score).

abduce_atom(Atom, Abd, Abd, Score1, Score2) :-
% the terminate state introduce no abduced atoms, only check if the last position is my_goal/1.
Atom =.. [terminate, X, []],
abduce_terminate(X, [], Abd, Abd, Score1, Score2).

%% abduce movements in four directions
abduce_up([A,B|T], [B|T], Old_abd_facts, New_abd_facts, Old_score, New_score) :-
    abduce_nn(A, [X0,Y0], Old_abd_facts, Inter_abd_facts, Old_score, Inter_score),
    abduce_nn(B, [X0,Y1], Inter_abd_facts, New_abd_facts, Inter_score, New_score),
%%    is_new_position([X0,Y1],Inter_abd_facts),
    height(H),
    Y0 < H,
    Y1 is Y0 + 1.

abduce_down([A,B|T], [B|T], Old_abd_facts, New_abd_facts, Old_score, New_score) :-
    abduce_nn(A, [X0,Y0], Old_abd_facts, Inter_abd_facts, Old_score, Inter_score),
    abduce_nn(B, [X0,Y1], Inter_abd_facts, New_abd_facts, Inter_score, New_score),
%    is_new_position([X0,Y1],Inter_abd_facts),
    Y0 > 0,
    Y1 is Y0 - 1.

abduce_left([A,B|T], [B|T], Old_abd_facts, New_abd_facts, Old_score, New_score) :-
    abduce_nn(A, [X0,Y0], Old_abd_facts, Inter_abd_facts, Old_score, Inter_score),
    abduce_nn(B, [X1,Y0], Inter_abd_facts, New_abd_facts, Inter_score, New_score),
%    is_new_position([X0,Y1],Inter_abd_facts),
    X0 > 0,
    X1 is X0 - 1.

abduce_right([A,B|T], [B|T], Old_abd_facts, New_abd_facts, Old_score, New_score) :-
    abduce_nn(A, [X0,Y0], Old_abd_facts, Inter_abd_facts, Old_score, Inter_score),
    abduce_nn(B, [X1,Y0], Inter_abd_facts, New_abd_facts, Inter_score, New_score),
    width(W),
    X0 < W,
    X1 is X0 + 1.

abduce_terminate([A], [], Abd_facts, Abd_facts, Old_score, New_score) :-
    abduce_nn(A, [X,Y], Abd_facts, Abd_facts, Old_score, New_score),
    my_goal([X,Y]).

%% Abduction with probabilistic facts

abduce_nn(A, [X,Y], Abd_facts, Abd_facts, Placeholder_score, Placeholder_score) :-
    %% previously abduced
    member(mario_at(A, [X,Y]), Abd_facts).

abduce_nn(A, [X,Y], Abd, [mario_at(A, [X,Y])|Abd], Old_score, New_score) :-
    nn(A, [X,Y], Prob),
    ic(A, Abd),
    New_score is Old_score*Prob.

%% Integrity constraints
% Mario can only be in one place it one picture
ic(A, Abduced) :-
    findall(Pic, member(mario_at(Pic, _), Abduced), Pics),
    \+member(A, Pics).
%is_new_position(Pos, Abduced) :-
%    findall(P, member(mario_at(_, P), Abduced), Positions),
%    \+member(Pos, Positions).