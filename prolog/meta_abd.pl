% This file is a copyrighted under the BSD 3-clause licence, details of which can be found in the root directory.
% Code for Generating by Understanding: Neural Visual Generation with Logical Symbol Groundings
% https://arxiv.org/abs/2310.17451

:- module(meta_abd,[metaabd/2, learn/2,learn/5,learn_seq/2,pprint/1,op(950,fx,'@')]).

:- dynamic
    ibk/3,
    functional/0,
    user:head_pred/1,
    user:body_pred/1,
    user:abducible/1,
    body_pred_call/2,
    abducible_call/6,
    compiled_pred_call/2.

default(max_clauses(10)).

metaabd(Pos,Neg):-
    nb_setval(n_vars, 0), %% number of new vars in constraints
    set_best_prob(-1),
    set_best_abd_prob(-1),
    set_best_program(false, []),
    learn(Pos,Neg,Prog,Abd,Score1),
    Abd \= [],
    prog_prob(Prog,Score2),
    Score is Score1 * Score2,
    is_better(Score),
    format('% is better: ~f\n',[Score]),
    pprint(Prog),
    writeln(Abd),
    writeln(Score1),
    writeln(Score2),
    writeln(Abd),
    writeln('==========='),
    set_best_prob(Score),
    set_best_abd_prob(Score1),
    set_best_program(Prog,Abd),
    %% if no need for abduction, then terminate 
    (   Score1 < 1.0 -> false; metaabd_terminate).
metaabd(_,_):-
    metaabd_terminate.
metaabd_terminate :-
    get_best_program(Prog,Abd),
    writeln('- Program:'),
    pprint(Prog),
    writeln('- Abduced Facts:'),
    (   Abd \= [] -> writeln(Abd); true).

prog_prob(Prog,Score) :-
    once(complexity(Prog, [], N, M)),
    Score is 6/((N+M)*pi)^2.

complexity([], Subs, 0, M) :-
    list_to_set(Subs, Subs1), %% distinct predicates
    length(Subs1, M).
complexity([sub(Name,_P,_A,Subs)|Clauses], Subs1, N, M) :-
    append(Subs1, Subs, Subs2),
    metarule(Name,_Subs,_Atom,Body,_Recursive,_),
    length(Body, L),
    complexity(Clauses, Subs2, N1, M),
    N is L + N1.

set_best_prob(S):-
    nb_setval(best_prob,S).
get_best_prob(S):-
    nb_getval(best_prob,S).
set_best_abd_prob(S):-
    nb_setval(best_abd_prob,S).
get_best_abd_prob(S):-
    nb_getval(best_abd_prob,S).
is_better(S):-
    get_best_prob(Best),
    S > Best.

set_best_program(G,A):-
    nb_setval(best_program,G),
    nb_setval(best_abd,A).
get_best_program(G,A):-
    nb_getval(best_program,G),
    nb_getval(best_abd,A).

learn(Pos,Neg):-
    set_best_prob(-1),
    set_best_abd_prob(-1),
    set_best_program(false, []),
    learn(Pos,Neg,Prog,_Abd,_Score),
    pprint(Prog).

learn(Pos1,Neg1,Prog,Abd,Score):-
    setup,
    make_atoms(Pos1,Pos2),
    make_atoms(Neg1,Neg2),
    proveall(Pos2,Sig,Prog,Abd,Score),
    nproveall(Neg2,Sig,Prog,Abd,Score),
    ground(Prog),
    ground(Abd),
    check_functional(Pos2,Sig,Prog,Abd,Score).

learn(_,_,_,_):-!,
    writeln('% unable to learn a solution'),
    false.

proveall(Atoms,Sig,Prog,Abd,Score):-
    target_predicate(Atoms,P/A),
    format('% learning ~w\n',[P/A]),
    iterator(MaxN),
    format('% clauses: ~d\n',[MaxN]),
    invented_symbols(MaxN,P/A,Sig),
    assert_sig_types(Sig),
    prove_examples(Atoms,Sig,_Sig,MaxN,0,_N,[],Prog,[],Abd,1.0,Score).

prove_examples([],_FullSig,_Sig,_MaxN,N,N,Prog,Prog,Abd,Abd,S,S).
prove_examples([Atom|Atoms],FullSig,Sig,MaxN,N1,N2,Prog1,Prog2,Abd1,Abd2,S1,S2):-
    deduce_atom(Atom,FullSig,Prog1,Abd1,S1),!,
    check_functional([Atom],Sig,Prog1,Abd1,S1),
    prove_examples(Atoms,FullSig,Sig,MaxN,N1,N2,Prog1,Prog2,Abd1,Abd2,S1,S2).
prove_examples([Atom|Atoms],FullSig,Sig,MaxN,N1,N2,Prog1,Prog2,Abd1,Abd2,S1,S2):-
    prove([Atom],FullSig,Sig,MaxN,N1,N3,Prog1,Prog3,Abd1,Abd3,S1,S3),
    check_functional([Atom],Sig,Prog3,Abd1,S1),
    prove_examples(Atoms,FullSig,Sig,MaxN,N3,N2,Prog3,Prog2,Abd3,Abd2,S3,S2).

deduce_atom(Atom,Sig,Prog,Abd,S):-
    length(Prog,N),
    prove([Atom],Sig,_,N,N,N,Prog,Prog,Abd,Abd,S,S).

prove([],_FullSig,_Sig,_MaxN,N,N,Prog,Prog,Abd,Abd,S,S).
prove([Atom|Atoms],FullSig,Sig,MaxN,N1,N2,Prog1,Prog2,Abd1,Abd2,S1,S2):-
    prove_aux(Atom,FullSig,Sig,MaxN,N1,N3,Prog1,Prog3,Abd1,Abd3,S1,S3),
    prove(Atoms,FullSig,Sig,MaxN,N3,N2,Prog3,Prog2,Abd3,Abd2,S3,S2).

prove_aux('@'(Atom),_FullSig,_Sig,_MaxN,N,N,Prog,Prog,Abd,Abd,S,S):- !,
    user:call(Atom).

prove_aux(p(P,A,Args,_Path),_FullSig,_Sig,_MaxN,N,N,Prog,Prog,Abd,Abd,S,S):-
    nonvar(P),
    type(P,A,compiled_pred),!,
    compiled_pred_call(P,Args).

prove_aux(p(P,A,Args,_Path),_FullSig,_Sig,_MaxN,N,N,Prog,Prog,Abd,Abd,S,S):-
    (   nonvar(P) -> type(P,A,body_pred); true),
    body_pred_call(P,Args).
%% abduce incomplete background knowledge
prove_aux(p(P,A,Args,_Path),_FullSig,_Sig,_MaxN,N,N,Prog,Prog,Abd1,Abd2,S1,S2):-
    (   nonvar(P) ->
        (   type(P,A,abducible),
            type(P,A,body_pred),
            \+body_pred_call(P,Args))
    ;   true),
    abducible_call(P,Args,Abd1,Abd2,S1,S2),
    %% writeln(Abd2),
    get_best_abd_prob(Max_Prob),
    S2 > Max_Prob.

prove_aux(p(P,A,Args,Path),FullSig,Sig,MaxN,N1,N2,Prog1,Prog2,Abd1,Abd2,S1,S2):-
    (   var(P) -> true; (\+ type(P,A,head_pred), !, type(P,A,ibk_head_pred))),
    ibk([P|Args],Body,Path),
    prove(Body,FullSig,Sig,MaxN,N1,N2,Prog1,Prog2,Abd1,Abd2,S1,S2).

prove_aux(p(P,A,Args,Path),FullSig,Sig1,MaxN,N1,N2,Prog1,Prog2,Abd1,Abd2,S1,S2):-
    N1 \== 0,
    Atom = [P|Args],
    select_lower(P,A,FullSig,Sig1,Sig2),
    member(sub(Name,P,A,Subs),Prog1),
    metarule(Name,Subs,Atom,Body,Recursive,[Atom|Path]),
    check_recursion(Recursive,MaxN,Atom,Path),
    prove(Body,FullSig,Sig2,MaxN,N1,N2,Prog1,Prog2,Abd1,Abd2,S1,S2).

prove_aux(p(P,A,Args,Path),FullSig,Sig1,MaxN,N1,N2,Prog1,Prog2,Abd1,Abd2,S1,S2):-
    N1 \== MaxN,
    Atom = [P|Args],
    bind_lower(P,A,FullSig,Sig1,Sig2),
    metarule(Name,Subs,Atom,Body,Recursive,[Atom|Path]), % ??
    check_recursion(Recursive,MaxN,Atom,Path),
    check_new_metasub(Name,P,A,Subs,Prog1),
    succ(N1,N3),
    prove(Body,FullSig,Sig2,MaxN,N3,N2,[sub(Name,P,A,Subs)|Prog1],Prog2,Abd1,Abd2,S1,S2).

nproveall(Atoms,Sig,Prog,Abd,Score):-
    forall(member(Atom,Atoms), \+ deduce_atom(Atom,Sig,Prog,Abd,Score)).

make_atoms(Atoms1,Atoms2):-
    maplist(atom_to_list,Atoms1,Atoms3),
    maplist(make_atom,Atoms3,Atoms2).

make_atom([P|Args],p(P,A,Args,[])):-
    length(Args,A).

check_functional(Atoms,Sig,Prog,Abd,Score):-
    (   functional ->
        forall(member(Atom1,Atoms),
               \+ (
                    make_atom(Atom2,Atom1),
                    user:func_test(Atom2,TestAtom2,Condition),
                    make_atom(TestAtom2,TestAtom1),
                    deduce_atom(TestAtom1,Sig,Prog,Abd,Score),
                    \+ call(Condition)));
        true).

check_recursion(false,_,_,_).
check_recursion(true,MaxN,Atom,Path):-
    MaxN \== 1, % need at least two clauses if we are using recursion
    \+memberchk(Atom,Path).

select_lower(P,A,FullSig,_Sig1,Sig2):-
    nonvar(P),!,
    append(_,[sym(P,A,_)|Sig2],FullSig),!.
select_lower(P,A,_FullSig,Sig1,Sig2):-
    append(_,[sym(P,A,U)|Sig2],Sig1),
    (var(U)-> !,fail;true ).

bind_lower(P,A,FullSig,_Sig1,Sig2):-
    nonvar(P),!,
    append(_,[sym(P,A,_)|Sig2],FullSig),!.
bind_lower(P,A,_FullSig,Sig1,Sig2):-
    append(_,[sym(P,A,U)|Sig2],Sig1),
    (var(U)-> U = 1,!;true).

check_new_metasub(Name,P,A,Subs,Prog):-
    memberchk(sub(Name,P,A,_),Prog),!,
    last(Subs,X),
    when(ground(X), \+memberchk(sub(Name,_P,A,Subs),Prog)).
check_new_metasub(_Name,_P,_A,_Subs,_Prog).

assert_sig_types(Sig):-
    forall((member(sym(P,A,_),Sig),\+type(P,A,head_pred)),
        assert(type(P,A,head_pred))).

head_preds:-
    %% remove old invented predicates (not that it really matters)
    retractall(type(_,_,head_pred)),
    forall((user:head_pred(P/A),\+type(P,A,head_pred)),
        assert(type(P,A,head_pred))).

body_preds:-
    retractall(type(_,_,body_pred)),
    retractall(body_pred_call(P,_)),
    findall(P/A,user:body_pred(P/A),S0),
    assert_body_preds(S0).

assert_body_preds(S1):-
    forall(member(P/A,S1),(
                           retractall(type(P,A,body_pred)),
                           retractall(body_pred_call(P,_)),
                           retractall(user:body_pred(P/A)),
                           (   current_predicate(P/A) -> (
                                                           assert(type(P,A,body_pred)),
                                                           assert(user:body_pred(P/A)),
                                                           functor(Atom,P,A),
                                                           Atom =.. [P|Args],
                                                           assert((body_pred_call(P,Args):-user:Atom))
                                                         );
                               format('% WARNING: ~w does not exist\n',[P/A])
                           )
                          )).
%
abducible_preds:-
    retractall(type(_,_,abducible)),
    retractall(abducible_call(P,_,_,_,_,_)),
    findall(P/A,(user:abducible(P/A),
                 user:body_pred(P/A)),
            S0),
    assert_abducibles(S0).

assert_abducibles(S1):-
    forall(member(P/A,S1),(
                           retractall(type(P,A,abducible)),
                           retractall(abducible_call(P,_,_,_,_,_)),
                           retractall(user:abducible(P/A)),
                           (   current_predicate(P/A) -> (
                                                           assert(type(P,A,abducible)),
                                                           assert(user:abducible(P/A)),
                                                           functor(Atom,P,A),
                                                           Atom =.. [P|Args],
                                                           assert((abducible_call(P,Args,Abd1,Abd2,Pr1,Pr2):-
                                                                  user:abduce(Atom,Abd1,Abd2,Pr1,Pr2)))
                                                         );
                               format('% WARNING: ~w does not exist\n',[P/A])
                           )
                          )).


ibk_head_preds:-
    findall(P/A,type(P,A,ibk_head_pred),S0),
    list_to_set(S0,S1),
    retractall(type(_,_,ibk_head_pred)),
    forall(member(P/A,S1),
        assert(type(P,A,ibk_head_pred))
    ).

compiled_preds:-
    findall(P/A,(type(P,A,compiled_pred), not(type(P,A,body_pred)), not(type(P,A,head_pred)), not(type(P,A,ibk_head_pred))),S0),
    list_to_set(S0,S1),
    retractall(type(_,_,compiled_pred)),
    retractall(compiled_pred_call(_,_)),
    forall(member(P/A,S1),(
        (current_predicate(P/A) -> (
            assert(type(P,A,compiled_pred)),
            functor(Atom,P,A),
            Atom =..[P|Args],
            assert((compiled_pred_call(P,Args):-user:Atom))
        );
            format('% WARNING: ~w does not exist\n',[P/A])
        )
    )).

options:-
    (current_predicate(min_clauses/1) -> true; (set_option(min_clauses(1)))),
    (current_predicate(max_clauses/1) -> true; (default(max_clauses(MaxN)),set_option(max_clauses(MaxN)))),
    (current_predicate(max_inv_preds/1) -> true; (max_clauses(MaxN),succ(MaxInv,MaxN),set_option(max_inv_preds(MaxInv)))).

set_option(Option):-
    functor(Option,Name,Arity),
    functor(Retract,Name,Arity),
    retractall(Retract),
    assert(Option).

setup:-
    options,
    head_preds,
    body_preds,
    abducible_preds,
    ibk_head_preds,
    compiled_preds.

iterator(N):-
    min_clauses(MinN),
    max_clauses(MaxN),
    between(MinN,MaxN,N).

target_predicate([p(P,A,_Args,[])|_],P/A).

%% target_predicates(Atoms, Preds2):-
%%     findall(P/A, member([p(inv,P,A,_Args,_Atom,[])],Atoms), Preds1),
%%     list_to_set(Preds1,Preds2).

invented_symbols(MaxClauses,P/A,[sym(P,A,_U)|Sig]):-
    NumSymbols is MaxClauses-1,
    max_inv_preds(MaxInvPreds),
    M is min(NumSymbols,MaxInvPreds),
    findall(sym(Sym1,_Artiy,_Used1),(between(1,M,I),atomic_list_concat([P,'_',I],Sym1)),Sig).

pprint(Prog1):-
    reverse(Prog1,Prog3),
    maplist(metasub_to_clause,Prog3,Prog2),
    maplist(pprint_clause,Prog2).

pprint_clause(C):-
    numbervars(C,0,_),
    format('~q.~n',[C]).

pprint_to_prog(Prog1, Prog2) :-
    reverse(Prog1,Prog3),
    maplist(metasub_to_clause,Prog3,Prog2),
    maplist(pprint_name_vars,Prog2).
pprint_name_vars(C):-
    numbervars(C,0,_).

metasub_to_clause(sub(Name,_,_,Subs),Clause2):-
    metarule(Name,Subs,HeadList,BodyAsList1,_,_),
    add_path_to_body(BodyAsList3,_,BodyAsList1),
    include(no_ordering,BodyAsList3,BodyAsList2),
    maplist(atom_to_list,ClauseAsList,[HeadList|BodyAsList2]),
    list_to_clause(ClauseAsList,Clause1),
    (   Clause1 = (H,T) -> Clause2=(H:-T); Clause2=Clause1).

no_ordering(H):-
    H\='@'(_).

list_to_clause([Atom],Atom):-!.
list_to_clause([Atom|T1],(Atom,T2)):-!,
    list_to_clause(T1,T2).

atom_to_list(Atom,AtomList):-
    Atom =..AtomList.

%% build the internal metarule clauses
user:term_expansion(metarule(Subs,Head,Body),Asserts):-
    metarule_asserts(_Name,Subs,Head,Body,_MetaBody,Asserts).
user:term_expansion(metarule(Name,Subs,Head,Body),Asserts):-
    metarule_asserts(Name,Subs,Head,Body,_MetaBody,Asserts).
user:term_expansion((metarule(Subs,Head,Body):-MetaBody),Asserts):-
    metarule_asserts(_Name,Subs,Head,Body,MetaBody,Asserts).
user:term_expansion((metarule(Name,Subs,Head,Body):-MetaBody),Asserts):-
    metarule_asserts(Name,Subs,Head,Body,MetaBody,Asserts).

metarule_asserts(Name,Subs,Head,Body1,MetaBody,[meta_abd:MRule]):-
    Head = [P|_],
    is_recursive(Body1,P,Recursive),
    add_path_to_body(Body1,Path,Body2),
    gen_metarule_id(Name,AssertName),
    %% very hacky - I assert that all ground body predicates are compiled
    %% I filter these in the setup call
    forall((member(p(P1,A1,_,_),Body2), ground(P1)), assert(type(P1,A1,compiled_pred))),
    (var(MetaBody) ->
        MRule = metarule(AssertName,Subs,Head,Body2,Recursive,Path);
        MRule = (metarule(AssertName,Subs,Head,Body2,Recursive,Path):-MetaBody)).

user:term_expansion((ibk(Head,Body):-IbkBody),(ibk(Head,Body):-IbkBody)):-
    ibk_asserts(Head,Body,IbkBody,[]).

user:term_expansion(ibk(Head,Body),ibk(Head,Body)):-
    ibk_asserts(Head,Body,false,[]).

ibk_asserts(Head,Body1,IbkBody,[]):-
    Head = [P0|Args1],
    length(Args1,A0),
    assert(type(P0,A0,ibk_head_pred)),
    add_path_to_body(Body1,Path,Body2),
    (IbkBody == false -> assert(ibk(Head,Body2,Path)); assert((ibk(Head,Body2,Path):-IbkBody))),
    %% very hacky - I assert that all ground body predicates are compiled
    %% I filter these in the setup call
    forall((member(p(P1,A1,_,_),Body2), ground(P1)), assert(type(P1,A1,compiled_pred))).

is_recursive([],_,false).
is_recursive([[Q|_]|_],P,true):-
    Q==P,!.
is_recursive([_|T],P,Res):-
    is_recursive(T,P,Res).

add_path_to_body([],_Path,[]).
add_path_to_body(['@'(Atom)|Atoms],Path,['@'(Atom)|Rest]):-
    add_path_to_body(Atoms,Path,Rest).
add_path_to_body([[P|Args]|Atoms],Path,[p(P,A,Args,Path)|Rest]):-
    length(Args,A),
    add_path_to_body(Atoms,Path,Rest).

gen_metarule_id(Name,Name):-
    ground(Name),!.
gen_metarule_id(_Name,IdNext):-
    current_predicate(metarule_next_id/1),!,
    metarule_next_id(Id),
    succ(Id,IdNext),
    set_option(metarule_next_id(IdNext)).
gen_metarule_id(_Name,1):-
    set_option(metarule_next_id(2)).

learn_seq(Seq,Prog):-
    maplist(learn_task,Seq,Progs),
    flatten(Progs,Prog).

learn_task(Pos/Neg,Prog1):-
    learn(Pos,Neg,Prog1),!,
    maplist(metasub_to_clause,Prog1,Prog2),
    forall(member(Clause,Prog2),assert(user:Clause)),
    findall(P/A,(member(sub(_Name,P,A,_Subs),Prog1)),Preds),!,
    assert_body_preds(Preds).
learn_task(_,[]).