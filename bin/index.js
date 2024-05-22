#!/usr/bin/env node

import fs from 'fs';
import chalk from 'chalk';
import boxen from 'boxen';
import { colorize } from 'json-colorizer';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import spawn from 'await-spawn';
import { LambdaCloudAPI } from 'lambda-cloud-node-api';
import persist from 'node-persist';
import ora from 'ora';

import debuger from "debug";
const debug = debuger("trainer");

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const storage = persist.create({dir: './db'});
await storage.init();
const spinner = ora();

process.on ("SIGINT", async () => {
    spinner.stop();
    await terminate(argv);
    process.exit(1);
});

const argv = yargs(hideBin(process.argv))
    .usage('Usage: $0 command [options]\n' +
        'Example: $0 run --hf_account [string] --hf_dataset [string] --hf-input-model [string] --hf-output-model [string]')
    .config(JSON.parse(fs.readFileSync(process.cwd() + '/config/' + 'trainer.json')))
    .demandOption(['lc-ssh-identity-file','lc-api-key','lc-ssh-key-name','lc-instance-name-prefix','wandb-api-key','hf-access-token','hf-dataset', 'hf-input-model','hf-output-model'])
    .env('LAMBDA_CLOUD_SSH_KEY_NAME')
    .option('lc-ssh-key-name', {
        type: 'string',
        alias: 'lambda-cloud-ssh-key-name',
        describe: "The SSH key name used to access the server instance. i.e. See https://cloud.lambdalabs.com/ssh-keys e.g. Brendt's Macbook Pro"
    })
    .env('LAMBDA_CLOUD_SSH_IDENTITY_FILE')
    .option('lc-ssh-identity-file', {
        type: 'string',
        alias: 'lambda-cloud-ssh-identity-file',
        describe: 'The local file path of the SSH key uploaded to https://cloud.lambdalabs.com/ssh-keys e.g. ~/.ssh/lambda_cloudcloud'
    })
    .env('LAMBDA_CLOUD_API_KEY')
    .option('lc-api-key', {
        type: 'string',
        alias: 'lambda-cloud-api-key',
        describe: 'The API key used to launch and terminate instances on Lambda Cloud i.e. https://cloud.lambdalabs.com/api-keys'
    })
    .option('lc-instance-name-prefix', {
        type: 'string',
        alias: 'lambda-cloud-instance-name-prefix',
        describe: 'The Lambda Cloud instance type to launch.',
        default: 'gpu_1x_a100_sxm4'
    })
    .option('lc-instance-type', {
        type: 'string',
        alias: 'lambda-cloud-instance-type',
        describe: 'The Lambda Cloud instance type to launch.',
        default: 'gpu_1x_a100_sxm4'
    })
    .option('lc-region', {
        type: 'string',
        alias: 'lambda-cloud-region',
        describe: 'The Lambda Cloud region in which to launch.',
        default: 'us-west-2'
    })
    .env('WANDB_API_KEY')
    .option('wandb-api-key', {
        type: 'string',
        alias: 'weights-and-biases-api-key',
        describe: 'Your Weights & Biases API Key e.g. https://wandb.ai/quickstart'
    })
    .env('HF_ACCESS_TOKEN')
    .option('hf-access-token', {
        type: 'string',
        alias: 'hugging-face-access-token',
        describe: 'Your Hugging Face User Access Token e.g. https://huggingface.co/docs/hub/en/security-tokens'
    })
    .option('hf-account', {
        type: 'string',
        alias: 'hugging-face-account',
        describe: 'The Hugging Face account from which the dataset is pulled and fine-tuned model is pushed. e.g. CodeHeroes',
        default: 'CodeHeroes'
    })
    .option('hf-dataset', {
        type: 'string',
        alias: 'hugging-face-dataset',
        describe: 'The dataset used to fine-tuning the base model. e.g. apple_app_store_training_data'
    })
    .option('hf-input-model', {
        type: 'string',
        alias: 'hugging-face-input-model',
        describe: 'The model repo from which the base model is pulled. e.g. unsloth/llama-3-8b-Instruct-bnb-4bit'
    })
    .option('hf-output-model', {
        type: 'string',
        alias: 'hugging-face-output-model',
        describe: 'The model repo that training results are pushed. e.g. llama-3-8b-Instruct-bnb-4bit'
    })
    .option('terminate', {
        type: "boolean",
        describe: 'Should the Lambda Cloud instance be terminated when training completes.',
        default: true
    })
    .command({
        command: 'launch',
        desc: 'Launch an instance on Lambda Cloud.',
        handler: (argv) => {
            launch(argv);
        }
    })
    .command({
        command: 'transfer',
        desc: 'Transfer files to the instance.',
        handler: (argv) => {
            transfer(argv);
        }
    })
    .command({
        command: 'config',
        desc: 'Apply server-side configuration.',
        handler: (argv) => {
            config(argv);
        }
    })
    .command({
        command: 'install',
        desc: 'Install server-side dependancies.',
        handler: (argv) => {
            install(argv);
        }
    })
    .command({
        command: 'authenticate',
        desc: 'Authenticate with Hugging Face (https://huggingface.co/) and Weights and Biases (https://wandb.ai/).',
        handler: (argv) => {
            authenticate(argv);
        }
    })
    .command({
        command: 'train',
        desc: 'Train the model.',
        handler: (argv) => {
            train(argv);
        }
    })
    .command({
        command: 'terminate',
        desc: 'Terminate the instance.',
        handler: (argv) => {
            terminate(argv);
        }
    })
    .command({
        command: 'run',
        desc: 'Run all commands in squence i.e. launch > transfer > config > install > authenticate > train > terminate',
        handler: (argv) => {
            run(argv);
        }
    })
    .middleware(function (argv) {
        debug(debug, argv);
    }, true)
    .help('help')
    .parse();

async function run(argv){
    await launch(argv)
    .then(() => transfer(argv))
    .then(() => config(argv))
    .then(() => install(argv))
    .then(() => authenticate(argv))
    .then(() => train(argv))
    .then(() => {if(argv.terminate == true) terminate(argv)})
    .catch((error) => {
        debug(error);
    });
};

async function launch(argv){
    const postfix = (Math.random() + 1).toString(36).substring(6);
    const instanceName = argv.lcInstanceNamePrefix + '-' + postfix;
    await storage.setItem('instanceName', instanceName)
    console.log(boxen(
        "Launching:\n" +
        "- type: " + chalk.green(argv.lcInstanceType) + '\n' +
        "- region: " + chalk.green(argv.lcRegion) + '\n' +
        "- name: " + chalk.green(instanceName) + '\n' + 
        "- auto-shutdown: " + chalk.green(argv.terminate ? 'yes' : 'no'),
        {padding: 1, borderColor: 'green', dimBorder: true}));

    const client = new LambdaCloudAPI({ apiKey: argv.lambdaCloudApiKey });
    await client.launchInstance({
        instance_type_name: argv.lcInstanceType,
        region_name: argv.lcRegion,
        ssh_key_names: [argv.lcSshKeyName],
        name: instanceName,
    })
    .then(async(launchedInstance) => {
        debug('Launched:\n' + colorize(JSON.stringify(launchedInstance, null, 4)));
        const instanceId = launchedInstance['instance_ids'][0];
        await storage.setItem('instanceId', instanceId)
        console.log('Launched: ' + chalk.green(await storage.getItem('instanceId')));
        await getInstanceIP(instanceId);
        debug('Host IP: ' + chalk.green(await storage.getItem('host')));
        return argv;
    })
    .catch((e) => {
        debug(e);
        const error = e['error'];
        let message = error.message;
        if(error.suggestion)
            message += '\n' + error.suggestion;
        console.log(chalk.red('Error:\n' + message));
        throw e;
    });
};

async function getInstanceIP(instanceId){

    debug('Instance Id: ' + chalk.green(instanceId));
    const client = new LambdaCloudAPI({ apiKey: argv.lambdaCloudApiKey });
    const instance = await client.getRunningInstance(instanceId)
    .catch((e) => {
        debug(e);
        throw e;
    });

    if(spinner.isSpinning == false && instance.status != 'terminating'){
        spinner.color = 'green';
        spinner.start();
    }

    if('ip' in instance){
        debug('Instance:\n' + colorize(JSON.stringify(instance, null, 4)));
        await storage.setItem('host', instance['ip']);
        debug('IP: ' + await storage.getItem('host'));
    }

    debug(instance.status);
    if(instance.status == 'active'){ 
        spinner.succeed('running');
        return;
    } else {
        spinner.text = instance.status + '...';
        await sleep(2000);
        await getInstanceIP(instanceId);
    }
}

async function transfer(argv){
    const ip = await storage.getItem('host');
    console.log(boxen(
        "Transfering files:\n" +
        "- to host: " + chalk.green(ip),
        {padding: 1, borderColor: 'green', dimBorder: true}));
    
    const process = spawn(
        'sh', 
        ['./training/file_transfer.sh'], 
        { stdio: 'inherit', shell: true, env: { HOST: ip, SSH_PUB_KEY: argv.lcSshIdentityFile } }
    );
    process.child.on('close', (code) => {
        return argv;
    });
    process.child.on('error', (error) => {
        debug(error);
        throw error;
    });
    await process;
};

async function config(argv){
    const ip = await storage.getItem('host');
    console.log(boxen(
        "Configuring:\n" +
        "- host: " + chalk.green(ip),
        {padding: 1, borderColor: 'green', dimBorder: true}));
        
    const process = spawn(
        'ssh', 
        ['-i ' + argv.lcSshIdentityFile, '-o StrictHostKeyChecking=no', 'ubuntu@' + ip, 'sudo ./config.sh'], 
        { stdio: 'inherit', shell: true }
    );
    process.child.on('close', () => {
        return argv;
    });
    process.child.on('error', (error) => {
        debug(error);
        throw error;
    });
    await process;
};

async function install(argv){
    const ip = await storage.getItem('host');
    console.log(boxen(
        "Installing dependancies:\n" +
        "- on host: " + chalk.green(ip),
        {padding: 1, borderColor: 'green', dimBorder: true}));
    
    const process = spawn(
        'ssh', 
        ['-i ' + argv.lcSshIdentityFile, '-o StrictHostKeyChecking=no', 'ubuntu@' + ip, './install.sh'], 
        { stdio: 'inherit', shell: true }
    );
    process.child.on('close', (code) => {
        return argv;
    });
    process.child.on('error', (error) => {
        debug(error);
        throw error;
    });
    await process;
};

async function authenticate(argv){
    const ip = await storage.getItem('host');
    console.log(boxen(
        "Authenticating:\n" +
        "- host: " + chalk.green(ip) + "\n" +
        "with HF and W&B.",
        {padding: 1, borderColor: 'green', dimBorder: true}));
    
    const process = spawn(
        'ssh', 
        ['-i ' + argv.lcSshIdentityFile, '-o StrictHostKeyChecking=no', '-o SendEnv=HF_TOKEN', '-o SendEnv=WANDB_API_KEY', 'ubuntu@' + ip, './authenticate.sh'], 
        { stdio: 'inherit', shell: true, env: { HF_TOKEN:argv.hfAccessToken, WANDB_API_KEY:argv.wandbApiKey  } }
    );
    process.child.on('close', (code) => {
        return argv;
    });
    process.child.on('error', (error) => {
        debug(error);
        throw error;
    });
    await process;
};

async function train(argv){
    const ip = await storage.getItem('host');
    const instanceName = await storage.getItem('instanceName');
    console.log(boxen(
        "Training:\n" +
        "- on host: " + chalk.green(ip) + "\n" +
        "- run name: " + chalk.green(instanceName) + "\n" +
        "- with dataset: " + chalk.green(argv.hfAccount + "/" + argv.hfDataset) + "\n" +
        "- with model: " + chalk.green(argv.hfInputModel) + "\n" +
        "- push to model: " + chalk.green(argv.hfAccount + "/" + argv.hfOutputModel) + "\n" +
        "- will shutdown on completion: " + chalk.green(argv.terminate ? 'Yes' : 'No'), 
        {padding: 1, borderColor: 'green', dimBorder: true}));
    
    const process = spawn(
        'ssh', 
        ['-i ' + argv.lcSshIdentityFile, '-o StrictHostKeyChecking=no', '-o SendEnv=HF_TOKEN', '-o SendEnv=HF_ACCOUNT', '-o SendEnv=HF_DATASET', '-o SendEnv=HF_INPUT_MODEL', '-o SendEnv=HF_OUTPUT_MODEL', '-o SendEnv=WANDB_PROJECT', '-o SendEnv=WANDB_NAME', 'ubuntu@' + ip, 'python ./train.py'], 
        { stdio: 'inherit', shell: true, env: { HF_TOKEN: argv.hfAccessToken, HF_ACCOUNT: argv.hfAccount, HF_DATASET: argv.hfDataset, HF_INPUT_MODEL: argv.hfInputModel, HF_OUTPUT_MODEL: argv.hfOutputModel, WANDB_PROJECT: argv.hfOutputModel, WANDB_NAME: instanceName } }
    )
    process.child.on('close', (code) => {
        return argv;
    });
    process.child.on('error', (error) => {
        debug(error);
        throw error;
    });
    await process;
};

async function terminate(argv){
    const instanceId = await storage.getItem('instanceId')
    const ip = await storage.getItem('host')

    console.log(boxen(
        "Terminating:\n" +
        "- instance id: " + chalk.green(instanceId) + "\n" +
        "- with ip address: " + chalk.green(ip),
        {padding: 1, borderColor: 'green', dimBorder: true}));

    const client = new LambdaCloudAPI({ apiKey: argv.lambdaCloudApiKey });
    await client.terminateInstances([instanceId])
    .then((terminatedInstances) => {
        const instanceId = terminatedInstances['terminated_instances'][0].id;
        console.log('Terminated: ' + chalk.green(instanceId));
    })
    .catch((e) => {
        const error = e['error'];
        console.log(chalk.red('ERROR:\n' +
            error.message + '\n' +
            error.suggestion));
        throw error;
    });
};