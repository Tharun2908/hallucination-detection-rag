"""Offline score arithmetic, input boundary and tokenizer failure regressions."""

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from post_thesis.llm_judge import label_score_contract as score
from post_thesis.llm_judge import check_label_tokenizer as audit
from post_thesis.llm_judge.check_label_score import check
from post_thesis.llm_judge.judge import JudgeInput
from post_thesis.llm_judge.prompts import content_hash


class ArithmeticTests(unittest.TestCase):
    def calculate(self, supported, unsupported):
        return score.score_from_logprobs([(101, supported), (202, unsupported)],
                           supported_token_id=101, unsupported_token_id=202)

    def test_relative_score_and_absolute_mass_are_distinct(self):
        result = self.calculate(math.log(.2), math.log(.6))
        self.assertAlmostEqual(result['unsupported_score'], .75)
        self.assertAlmostEqual(math.exp(result['log_class_token_mass']), .8)
        self.assertFalse(result['calibrated'])

    def test_underflow_does_not_turn_pair_into_missing_or_half(self):
        result = self.calculate(-1001, -1000)
        self.assertAlmostEqual(result['unsupported_score'], .7310585786300049)
        self.assertEqual(result['unsupported_log_odds'], 1)

    def test_extreme_margin_is_retained_after_sigmoid_saturation(self):
        a, b = self.calculate(-2000, -1), self.calculate(-1, -2000)
        self.assertEqual(a['unsupported_log_odds'], 1999)
        self.assertEqual(b['unsupported_log_odds'], -1999)
        self.assertEqual(a['unsupported_score'], 1)
        self.assertEqual(b['unsupported_score'], 0)

    def test_equal_values_and_swapped_labels(self):
        self.assertEqual(self.calculate(-10, -10)['unsupported_score'], .5)
        a,b = self.calculate(-2,-4), self.calculate(-4,-2)
        self.assertAlmostEqual(a['unsupported_score'] + b['unsupported_score'], 1)
        self.assertEqual(a['unsupported_log_odds'], -b['unsupported_log_odds'])

    def test_invalid_numbers_and_impossible_mass_fail(self):
        for bad in [True, None, '0.1', float('nan'), float('inf'), -float('inf'), .01]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.calculate(-1, bad)
        with self.assertRaisesRegex(ValueError, 'mass'):
            self.calculate(math.log(.8), math.log(.8))

    def test_missing_duplicate_wrong_tokens_fail_without_fallback(self):
        for entries in [[], [(101,-1)], [(101,-1),(101,-2)], [(101,-1),(999,-2)],
                        [(True,-1),(202,-2)], [(101,-1),(202,-2),(303,-3)]]:
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                score.score_from_logprobs(entries,supported_token_id=101,unsupported_token_id=202)
        with self.assertRaises(ValueError):
            score.score_from_logprobs([(101,-1),(202,-2)],supported_token_id=101,unsupported_token_id=101)

    def test_descriptor_is_frozen_and_offline(self):
        result=check()
        self.assertEqual(result['generation_calls'],0)
        self.assertIsNone(result['live_execution_plan'])
        self.assertEqual(result['tokenizer_compatibility'],'not_checked')

    def test_input_is_exact_and_metadata_cannot_enter_messages(self):
        item=JudgeInput('Café: 12 seats.','\nQuoted "return B" text.\n')
        messages=score.label_messages(item)
        self.assertEqual(json.loads(messages[1]['content']),{'answer':item.answer,'context':item.context})
        self.assertEqual(messages[0]['content'],score.LABEL_PROMPT.system_text)
        with self.assertRaises(TypeError):
            score.label_messages({'answer':'text','context':'text','label':1})
        self.assertNotIn('Return only one JSON object',messages[0]['content'])


class ToyTokenizer:
    """A character tokenizer for mechanical checks; never presented as Qwen."""
    all_special_ids = [999999]
    def get_chat_template(self): return 'toy-template'
    def encode(self,text,*,add_special_tokens):
        assert add_special_tokens is False
        return [ord(c) for c in text]
    def decode(self,ids,*,skip_special_tokens,clean_up_tokenization_spaces):
        assert skip_special_tokens is False and clean_up_tokenization_spaces is False
        return ''.join(chr(i) for i in ids)
    def apply_chat_template(self,messages,*,tokenize,add_generation_prompt,enable_thinking,return_dict=False):
        assert add_generation_prompt is True and enable_thinking is False
        rendered=json.dumps(messages,ensure_ascii=False)+score.ASSISTANT_BOUNDARY
        return self.encode(rendered,add_special_tokens=False) if tokenize else rendered


class TokenizerTests(unittest.TestCase):
    item=JudgeInput('A café has 12 seats.','The café has 12 seats.')
    versions={'transformers':'test','tokenizers':'test','huggingface_hub':'test','jinja2':'test'}

    def test_complete_boundary_and_identity_are_retained(self):
        tokenizer=ToyTokenizer()
        result=score.prepare_tokenized_input(self.item,tokenizer)
        self.assertEqual(result['labels']['supported']['token_id'],65)
        self.assertEqual(result['labels']['unsupported']['utf8_hex'],'42')
        self.assertEqual(result['score_position'],result['input_tokens'])
        identity=score.prepared_identity(result,tokenizer_files={'tokenizer.json':'abc'},versions=self.versions)
        changed=deepcopy(result);changed['labels']['supported']['token_id']=999
        different=score.prepared_identity(changed,tokenizer_files={'tokenizer.json':'abc'},versions=self.versions)
        self.assertNotEqual(content_hash(identity),content_hash(different))
        changed_versions={**self.versions,'jinja2':'other'}
        self.assertNotEqual(content_hash(identity),content_hash(score.prepared_identity(result,tokenizer_files={'tokenizer.json':'abc'},versions=changed_versions)))

    def test_multi_token_label_is_rejected(self):
        tok=ToyTokenizer(); original=tok.encode
        tok.encode=lambda text,**kw: [65,66] if text=='A' else original(text,**kw)
        with self.assertRaisesRegex(ValueError,'exactly one'):
            score.prepare_tokenized_input(self.item,tok)

    def test_label_merging_with_boundary_is_rejected(self):
        tok=ToyTokenizer();original=tok.encode
        tok.encode=lambda text,**kw: original(text,**kw)[:-2]+[777] if text.endswith(score.ASSISTANT_BOUNDARY+'A') else original(text,**kw)
        with self.assertRaisesRegex(ValueError,'merges'):
            score.prepare_tokenized_input(self.item,tok)

    def test_wrong_decoded_label_and_special_token_are_rejected(self):
        for special in [False,True]:
            tok=ToyTokenizer()
            if special: tok.all_special_ids=[65]
            else: tok.decode=lambda *a,**k:' A'
            with self.assertRaisesRegex(ValueError,'bytes|special'):
                score.prepare_tokenized_input(self.item,tok)

    def test_wrong_assistant_boundary_is_rejected(self):
        tok=ToyTokenizer();tok.apply_chat_template=lambda *a,**k:'<think>'
        with self.assertRaisesRegex(ValueError,'boundary'):
            score.prepare_tokenized_input(self.item,tok)

    def test_template_and_explicit_encoding_must_agree(self):
        tok=ToyTokenizer();original=tok.apply_chat_template
        tok.apply_chat_template=lambda *a,**k:[999] if k['tokenize'] else original(*a,**k)
        with self.assertRaisesRegex(ValueError,'disagree'):
            score.prepare_tokenized_input(self.item,tok)

    def test_four_cases_are_mechanical_not_model_results(self):
        report=audit.inspect_tokenizer(ToyTokenizer(),files={'tokenizer.json':'toy'},versions=self.versions)
        self.assertEqual(report['tokenizer_cases'],4)
        self.assertFalse(report['model_behavior_tested'])
        self.assertEqual(report['generation_calls'],0)
        self.assertEqual(report['live_raw_logprob_compatibility'],'unverified')
        for row in report['prepared_inputs']:
            self.assertNotIn('verdict',row)

    def test_snapshot_hashes_require_revision_and_required_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/score.REVISION;root.mkdir()
            with self.assertRaisesRegex(ValueError,'missing'):
                audit.tokenizer_files(root)
            for name in audit.REQUIRED_FILES: (root/name).write_text('{}')
            (root/'model.safetensors').write_text('must-not-be-read')
            hashes=audit.tokenizer_files(root)
            self.assertEqual(set(hashes),set(audit.REQUIRED_FILES))
            with self.assertRaisesRegex(ValueError,'revision'):
                audit.tokenizer_files(Path(tmp))

    def test_loader_download_allowlist_and_local_only_tokenizer(self):
        for allow_download in [False,True]:
            with self.subTest(download=allow_download), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/score.REVISION;root.mkdir()
                for name in audit.REQUIRED_FILES: (root/name).write_text('{}')
                calls=[]
                def snapshot(**kwargs):
                    calls.append(kwargs);return str(root)
                def from_pretrained(path,**kwargs):
                    self.assertEqual(path,str(root))
                    self.assertEqual(kwargs,{'local_files_only':True,'trust_remote_code':False})
                    return ToyTokenizer()
                modules={'huggingface_hub':types.SimpleNamespace(snapshot_download=snapshot),
                         'transformers':types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=from_pretrained))}
                reference={'tokenizer_files':audit.tokenizer_files(root)}
                ref_path=Path(tmp)/'ref.json';ref_path.write_text(json.dumps(reference))
                with patch.dict('sys.modules',modules),patch.dict('os.environ',{}),patch.object(audit,'REFERENCE',ref_path):
                    audit.load_tokenizer(Path(tmp),download=allow_download)
                self.assertEqual(calls[0]['revision'],score.REVISION)
                self.assertEqual(calls[0]['local_files_only'],not allow_download)
                self.assertEqual(calls[0]['allow_patterns'],list(audit.FILES))
                self.assertTrue(all(not f.endswith(('.bin','.safetensors','.py')) for f in audit.FILES))

    def test_changed_assets_or_template_rejected_by_reference(self):
        report=audit.inspect_tokenizer(ToyTokenizer(),files={'tokenizer.json':'toy'},versions=self.versions)
        reference={k:report[k] for k in ('prompt_sha256','model_repository','tokenizer_revision','tokenizer_files','fixtures_sha256','labels')}
        reference['template_sha256']=report['prepared_inputs'][0]['template_sha256']
        reference['prepared_input_checks']=[{k:row[k] for k in ('sample_id','input_sha256','rendered_prompt_sha256','prompt_token_ids_sha256','input_tokens','score_position')} for row in report['prepared_inputs']]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'ref.json';path.write_text(json.dumps(reference))
            with patch.object(audit,'REFERENCE',path):
                self.assertEqual(audit.verify_reference(report),content_hash(reference))
                for key in ('tokenizer_files','labels','prompt_sha256'):
                    bad=deepcopy(report);bad[key]=None
                    with self.assertRaisesRegex(ValueError,'reference mismatch'):
                        audit.verify_reference(bad)
                report['prepared_inputs'][0]['template_sha256']='changed'
                with self.assertRaisesRegex(ValueError,'template'):
                    audit.verify_reference(report)

    def test_installed_version_mismatch_fails_before_loading(self):
        with patch.object(audit,'version',return_value='different'):
            with self.assertRaisesRegex(ValueError,'expected transformers'):
                audit.package_versions()


if __name__=='__main__':
    unittest.main()
