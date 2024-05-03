import * as vscode from 'vscode';
import axios from 'axios';

export function activate(context: vscode.ExtensionContext) {
    let disposable = vscode.commands.registerCommand('mathobjectidentifier.describeObject', async () => {
        const editor = vscode.window.activeTextEditor;

        if (!editor) {
            vscode.window.showErrorMessage('No open text editor.');
            return;
        }

        const selection = editor.selection;
        const text = editor.document.getText(selection);  // text instead of real prompt for testing

        if (!text) {
            vscode.window.showInformationMessage('No text selected.');
            return;
        }

        const apiKey = vscode.workspace.getConfiguration().get<string>('chatGPT.apiKey');
        if (!apiKey || apiKey.trim() === "") {
            vscode.window.showInformationMessage('Please configure your OpenAI API key in the Settings.');
            return;
        }

        const headers = {
            "Authorization": `Bearer ${apiKey}`,
            "Content-Type": "application/json"
        };
    
        const data = {
            "model": "gpt-3.5-turbo",
            "prompt": text,
            "temperature": 0.3
        };

        try {
            const response = await axios.post('https://api.openai.com/v1/completions', data, { headers: headers });
            if (response.data.choices && response.data.choices.length > 0) {
                vscode.window.showInformationMessage(response.data.choices[0].text);
            }
        } catch (error) {
            console.error('Error sending text to GPT:', error);
            vscode.window.showErrorMessage('Error sending text to GPT: ' + (error as Error).message);
        }
    });

    context.subscriptions.push(disposable);
}

export function deactivate() {}
