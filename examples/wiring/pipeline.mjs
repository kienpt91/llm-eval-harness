// Wire a Node pipeline to llm-eval through the `command` provider.
// One JSON request on stdin, one JSON answer on stdout. Replace the body with a call into your own code.
import { stdin, stdout } from "node:process";

let text = "";
for await (const chunk of stdin) text += chunk;
const { case: testCase } = JSON.parse(text); // {"protocol": 1, "case": {...}, "repeat": 0, "attempt": 1, "seed": null}

const raw = testCase.input;
const question = typeof raw === "object" && raw !== null ? raw.question : String(raw);
stdout.write(JSON.stringify({
  text: `You asked: ${question}`,
  tool_calls: [], // report EVERY side effect your pipeline performed or requested
  citations: (testCase.context ?? []).slice(0, 1).map((doc) => doc.id),
  usage: { model: "my-model", input_tokens: question.split(/\s+/).length, output_tokens: 5 },
}));
