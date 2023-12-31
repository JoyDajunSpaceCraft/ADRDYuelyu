/*
 * extension.ts
 *
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 *
 * Provides client for Pyright Python language server. This portion runs
 * in the context of the VS Code process and talks to the server, which
 * runs in another process.
 */

import * as path from 'path';
import { ExecutionPathProps } from 'pytea/service/executionPaths';
import { commands, ExtensionContext, extensions, OutputChannel, TextEditor, TextEditorEdit, Uri } from 'vscode';
import {
    CancellationToken,
    ConfigurationParams,
    ConfigurationRequest,
    DidChangeConfigurationNotification,
    HandlerResult,
    LanguageClient,
    LanguageClientOptions,
    ResponseError,
    ServerOptions,
    TransportKind,
} from 'vscode-languageclient/node';

import { isThenable } from 'pyright-internal/common/core';

import { FileBasedCancellationStrategy } from './cancellationUtils';
import { Commands } from './commands';
import { PathManager } from './pathTreeProvider';

let cancellationStrategy: FileBasedCancellationStrategy | undefined;

const pythonPathChangedListenerMap = new Map<string, string>();

export function activate(context: ExtensionContext) {
    cancellationStrategy = new FileBasedCancellationStrategy();

    const bundlePath = context.asAbsolutePath(path.join('dist', 'server.js'));
    const debugOptions = { execArgv: ['--nolazy', '--inspect=6600'] };

    // If the extension is launched in debug mode, then the debug server options are used.
    const serverOptions: ServerOptions = {
        run: { module: bundlePath, transport: TransportKind.ipc, args: cancellationStrategy.getCommandLineArguments() },
        // In debug mode, use the non-bundled code if it's present. The production
        // build includes only the bundled package, so we don't want to crash if
        // someone starts the production extension in debug mode.
        debug: {
            module: bundlePath,
            transport: TransportKind.ipc,
            args: cancellationStrategy.getCommandLineArguments(),
            options: debugOptions,
        },
    };

    // Options to control the language client
    const clientOptions: LanguageClientOptions = {
        // Register the server for python source files.
        documentSelector: [
            {
                language: 'python',
            },
        ],
        synchronize: {
            // Synchronize the setting section to the server.
            configurationSection: ['python', 'pytea'],
        },
        connectionOptions: { cancellationStrategy: cancellationStrategy },
        middleware: {
            // Use the middleware hook to override the configuration call. This allows
            // us to inject the proper "python.pythonPath" setting from the Python extension's
            // private settings store.
            workspace: {
                configuration: (
                    params: ConfigurationParams,
                    token: CancellationToken,
                    next: ConfigurationRequest.HandlerSignature
                ): HandlerResult<any[], void> => {
                    // Hand-collapse "Thenable<A> | Thenable<B> | Thenable<A|B>" into just "Thenable<A|B>" to make TS happy.
                    const result: any[] | ResponseError<void> | Thenable<any[] | ResponseError<void>> = next(
                        params,
                        token
                    );

                    // For backwards compatibility, set python.pythonPath to the configured
                    // value as though it were in the user's settings.json file.
                    const addPythonPath = (
                        settings: any[] | ResponseError<void>
                    ): Promise<any[] | ResponseError<any>> => {
                        if (settings instanceof ResponseError) {
                            return Promise.resolve(settings);
                        }

                        const pythonPathPromises: Promise<string | undefined>[] = params.items.map((item) => {
                            if (item.section === 'python') {
                                const uri = item.scopeUri ? Uri.parse(item.scopeUri) : undefined;
                                return getPythonPathFromPythonExtension(languageClient.outputChannel, uri, () => {
                                    // Posts a "workspace/didChangeConfiguration" message to the service
                                    // so it re-queries the settings for all workspaces.
                                    languageClient.sendNotification(DidChangeConfigurationNotification.type, {
                                        settings: null,
                                    });
                                });
                            }
                            return Promise.resolve(undefined);
                        });

                        return Promise.all(pythonPathPromises).then((pythonPaths) => {
                            pythonPaths.forEach((pythonPath, i) => {
                                // If there is a pythonPath returned by the Python extension,
                                // always prefer this over the pythonPath that uses the old
                                // mechanism.
                                if (pythonPath !== undefined) {
                                    settings[i].pythonPath = pythonPath;
                                }
                            });
                            return settings;
                        });
                    };

                    if (isThenable(result)) {
                        return result.then(addPythonPath);
                    }

                    return addPythonPath(result);
                },
            },
        },
    };

    // Create the language client and start the client.
    const languageClient = new LanguageClient('python', 'Pytea', serverOptions, clientOptions);
    const disposable = languageClient.start();

    // Push the disposable to the context's subscriptions so that the
    // client can be deactivated on extension deactivation.
    context.subscriptions.push(disposable);

    const restartCommand = Commands.restartServer;
    context.subscriptions.push(
        commands.registerCommand(restartCommand, (...args: any[]) => {
            languageClient.sendRequest<string>('workspace/executeCommand', {
                command: restartCommand,
                arguments: args,
            });
        })
    );

    const pathManager = new PathManager(context, {
        selectPath: (pathId) => {
            languageClient.sendRequest('workspace/executeCommand', {
                command: Commands.selectPath,
                arguments: [pathId],
            });
        },
    });

    const analyzeFileCommand = Commands.analyzeFile;
    context.subscriptions.push(
        commands.registerTextEditorCommand(
            analyzeFileCommand,
            (editor: TextEditor, edit: TextEditorEdit, ...args: any[]) => {
                const cmd = {
                    command: analyzeFileCommand,
                    arguments: [editor.document.uri.toString(), ...args],
                };
                languageClient
                    .sendRequest<ExecutionPathProps[]>('workspace/executeCommand', cmd)
                    .then(async (response) => {
                        pathManager.applyPathProps(response);
                        // window.showInformationMessage(response);
                    });
            }
        )
    );
}

export function deactivate() {
    if (cancellationStrategy) {
        cancellationStrategy.dispose();
        cancellationStrategy = undefined;
    }

    // Return undefined rather than a promise to indicate
    // that deactivation is done synchronously. We don't have
    // anything to do here.
    return undefined;
}

// The VS Code Python extension manages its own internal store of configuration settings.
// The setting that was traditionally named "python.pythonPath" has been moved to the
// Python extension's internal store for reasons of security and because it differs per
// project and by user.
async function getPythonPathFromPythonExtension(
    outputChannel: OutputChannel,
    scopeUri: Uri | undefined,
    postConfigChanged: () => void
): Promise<string | undefined> {
    try {
        const extension = extensions.getExtension('ms-python.python');
        if (!extension) {
            outputChannel.appendLine('Python extension not found');
        } else {
            if (extension.packageJSON?.featureFlags?.usingNewInterpreterStorage) {
                if (!extension.isActive) {
                    outputChannel.appendLine('Waiting for Python extension to load');
                    await extension.activate();
                    outputChannel.appendLine('Python extension loaded');
                }

                const execDetails = await extension.exports.settings.getExecutionDetails(scopeUri);
                let result: string | undefined;
                if (execDetails.execCommand && execDetails.execCommand.length > 0) {
                    result = execDetails.execCommand[0];
                }

                if (extension.exports.settings.onDidChangeExecutionDetails) {
                    installPythonPathChangedListener(
                        extension.exports.settings.onDidChangeExecutionDetails,
                        scopeUri,
                        postConfigChanged
                    );
                }

                if (!result) {
                    outputChannel.appendLine(`No pythonPath provided by Python extension`);
                } else {
                    outputChannel.appendLine(`Received pythonPath from Python extension: ${result}`);
                }

                return result;
            }
        }
    } catch (error) {
        outputChannel.appendLine(
            `Exception occurred when attempting to read pythonPath from Python extension: ${JSON.stringify(error)}`
        );
    }

    return undefined;
}

function installPythonPathChangedListener(
    onDidChangeExecutionDetails: (callback: () => void) => void,
    scopeUri: Uri | undefined,
    postConfigChanged: () => void
) {
    const uriString = scopeUri ? scopeUri.toString() : '';

    // No need to install another listener for this URI if
    // it already exists.
    if (pythonPathChangedListenerMap.has(uriString)) {
        return;
    }

    onDidChangeExecutionDetails(() => {
        postConfigChanged();
    });

    pythonPathChangedListenerMap.set(uriString, uriString);
}
