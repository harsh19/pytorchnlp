#!/usr/bin/env python
import re
import io
import argparse
from collections import defaultdict, namedtuple

Rule = namedtuple('Rule', 'lhs, rhs')
Terminal = namedtuple('Terminal', 'value')
Nonterminal = namedtuple('Nonterminal', 'value')
Subtree = namedtuple('Subtree', 'parent, children')

def ReadParseTree(parse_string):
  rule_matcher = re.compile('(\([^\(\)]+\))')
  placeholder_prefix = '_|+|_'
  nodes = {}
  id = 0
  while parse_string.startswith('('):
    match = rule_matcher.search(parse_string)
    assert match, 'no match!!!'
    parts = match.groups()[0].strip('()').split()
    for i in range(len(parts)):
      if parts[i].startswith(placeholder_prefix):
        parts[i] = nodes[parts[i]]
      elif i == 0:
        parts[i] = Nonterminal(value=parts[i])
      else:
        parts[i] = Terminal(value=parts[i])
    new_subtree = Subtree(parent=parts[0], children=parts[1:])
    nodes[placeholder_prefix+str(id)] = new_subtree
    parse_string = parse_string[:match.span()[0]] + placeholder_prefix + str(id) + parse_string[match.span()[1]:]
    id+=1
  assert parse_string.startswith(placeholder_prefix), 'parse string doesnt start with the placeholder'
  return new_subtree

def ExtractRulesFromSubtree(subtree):
  rules = []
  rhs = []
  for child in subtree.children:
    if type(child) is Terminal:
      rhs.append( child )
    else:
      rhs.append( child.parent )
      rules.extend( ExtractRulesFromSubtree(child) )
  rules.append( Rule(lhs=subtree.parent, rhs=rhs) )
  return rules

def CountRuleFrequencies(rules):
  freq = defaultdict(int)
  for rule in rules:
    assert type(rule.rhs) is list, 'rhs should be a list'
    rule_hash = [rule.lhs]
    rule_hash.extend(rule.rhs)
    rule_hash = tuple(rule_hash)
    freq[rule_hash] += 1
  return freq

def EvaluateParseTree(candidate_parse, reference_parse):
  # read candidate/reference rules
  candidate_rules = CountRuleFrequencies( ExtractRulesFromSubtree(candidate_parse) )
  reference_rules = CountRuleFrequencies( ExtractRulesFromSubtree(reference_parse) )
  # compute precision
  candidate_rules_count, unnormalized_precision = 0.0, 0.0
  for rule in candidate_rules:
    candidate_rules_count += candidate_rules[rule]
    unnormalized_precision += min(candidate_rules[rule], reference_rules[rule])
  # compute recall
  reference_rules_count, unnormalized_recall = 0.0, 0.0
  for rule in reference_rules:
    reference_rules_count += reference_rules[rule]
    unnormalized_recall += min(candidate_rules[rule], reference_rules[rule])
  return (unnormalized_precision, candidate_rules_count, unnormalized_recall, reference_rules_count)

def EvaluatePosTags(candidate_postags, reference_postags):  
  if len(candidate_postags) != len(reference_postags):
    return (0, len(reference_postags))
  correct = 0
  for i in range(len(candidate_postags)):
    if candidate_postags[i] == reference_postags[i]:
      correct += 1
  return (correct, len(reference_postags))

# parse arguments
argParser = argparse.ArgumentParser(description="evaluates parses and/or POS-taggings")
argParser.add_argument("--reference_parses_filename", type=str, default='')
argParser.add_argument("--candidate_parses_filename", type=str, default='')
argParser.add_argument("--reference_postags_filename", type=str, default='')
argParser.add_argument("--candidate_postags_filename", type=str, default='')
args = argParser.parse_args()

# evaluate parses
if args.reference_parses_filename and args.candidate_parses_filename:
  ref_file, candid_file = io.open(args.reference_parses_filename), io.open(args.candidate_parses_filename)
  precision_numerator, precision_denominator, recall_numerator, recall_denominator = 0.0, 0.0, 0.0, 0.0
  for (ref_parse_string, candid_parse_string) in zip(ref_file, candid_file):
    ref_tree = ReadParseTree(ref_parse_string.strip())
    candid_tree = ReadParseTree(candid_parse_string.strip())
    (unnormalized_precision, candidate_rules_count, unnormalized_recall, reference_rules_count) = \
        EvaluateParseTree(candid_tree, ref_tree)
    precision_numerator += unnormalized_precision
    precision_denominator += candidate_rules_count
    recall_numerator += unnormalized_recall
    recall_denominator += reference_rules_count
  ref_file.close()
  candid_file.close()
  precision = precision_numerator / precision_denominator
  recall = recall_numerator / recall_denominator
  f1 = 2 * precision * recall / (precision + recall)
  print('parsing precision = ', precision, ', recall = ', recall, ', f1 = ', f1)

# evaluate pos tags
if args.reference_postags_filename and args.candidate_postags_filename:
  ref_file, candid_file = io.open(args.reference_postags_filename), io.open(args.candidate_postags_filename)
  accuracy_numerator, accuracy_denominator = 0.0, 0.0
  for (ref_postags_string, candid_postags_string) in zip(ref_file, candid_file):
    ref_postags, candid_postags = ref_postags_string.split(), candid_postags_string.split()
    (correct, total) = EvaluatePosTags(candid_postags, ref_postags)
    accuracy_numerator += correct
    accuracy_denominator += total
  ref_file.close()
  candid_file.close()
  accuracy = accuracy_numerator / accuracy_denominator
  print('POS tagging accuracy = ', accuracy)
    
