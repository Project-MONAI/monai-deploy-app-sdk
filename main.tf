/*
export these variables before running this file
ARM_CLIENT_ID
ARM_SUBSCRIPTION_ID
ARM_TENANT_ID
ARM_CLIENT_SECRET
*/

# We strongly recommend using the required_providers block to set the
# Azure Provider source and version being used
terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "=3.0.0"
    }
  }
}

# Configure the Microsoft Azure Provider
provider "azurerm" {
  features {}
}

# Create a resource group
resource "azurerm_resource_group" "gh-actions-build-monai-models-resource-group" {
  name     = "gh-actions-build-monai-models-resource-group"
  location = "West Europe"
}

# Create a virtual network within the resource group
resource "azurerm_virtual_network" "gh-actions-build-monai-models-virtual-network" {
  name                = "gh-actions-build-monai-models-virtual-network"
  resource_group_name = azurerm_resource_group.gh-actions-build-monai-models-resource-group.name
  location            = azurerm_resource_group.gh-actions-build-monai-models-resource-group.location
  address_space       = ["10.0.0.0/16"]
}

resource "azurerm_subnet" "gh-actions-build-monai-models-internal-subnet" {
  name                 = "gh-actions-build-monai-models-internal-subnet"
  resource_group_name  = azurerm_resource_group.gh-actions-build-monai-models-resource-group.name
  virtual_network_name = azurerm_virtual_network.gh-actions-build-monai-models-virtual-network.name
  address_prefixes     = ["10.0.2.0/24"]
}

# Create public IPs
resource "azurerm_public_ip" "gh-actions-build-monai-models-public-ip" {
  name                = "gh-actions-build-monai-models-public-ip"
  location            = azurerm_resource_group.gh-actions-build-monai-models-resource-group.location
  resource_group_name = azurerm_resource_group.gh-actions-build-monai-models-resource-group.name
  allocation_method   = "Dynamic"
}

resource "azurerm_network_interface" "gh-actions-build-monai-models-network-interface" {
  name                = "gh-actions-build-monai-models-network-interface"
  location            = azurerm_resource_group.gh-actions-build-monai-models-resource-group.location
  resource_group_name = azurerm_resource_group.gh-actions-build-monai-models-resource-group.name

  ip_configuration {
    name                          = "gh-actions-build-monai-models-network-interface-ip-configuration"
    subnet_id                     = azurerm_subnet.gh-actions-build-monai-models-internal-subnet.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.gh-actions-build-monai-models-public-ip.id
  }
}

# Create Network Security Group and rule
resource "azurerm_network_security_group" "gh-actions-build-monai-models-nsg" {
  name                = "gh-actions-build-monai-models-nsg"
  location            = azurerm_resource_group.gh-actions-build-monai-models-resource-group.location
  resource_group_name = azurerm_resource_group.gh-actions-build-monai-models-resource-group.name

  security_rule {
    name                       = "SSH"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}

# Connect the security group to the network interface
resource "azurerm_network_interface_security_group_association" "gh-actions-build-monai-models-ga" {
  network_interface_id      = azurerm_network_interface.gh-actions-build-monai-models-network-interface.id
  network_security_group_id = azurerm_network_security_group.gh-actions-build-monai-models-nsg.id
}

resource "azurerm_linux_virtual_machine" "gh-actions-build-monai-models-vm" {
  name                = "gh-actions-build-monai-models-vm"
  resource_group_name = azurerm_resource_group.gh-actions-build-monai-models-resource-group.name
  location            = azurerm_resource_group.gh-actions-build-monai-models-resource-group.location
  // Standard_NC4as_T4_v3 has GPU. This has a cost associated!!!
  size           = "Standard_NC4as_T4_v3"
  admin_username = "adminuser"
  network_interface_ids = [
    azurerm_network_interface.gh-actions-build-monai-models-network-interface.id,
  ]

  admin_ssh_key {
    username   = "adminuser"
    public_key = file("/tmp/ssh_id_gh.pub") //This file is in the vm where you run terraform!!
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "StandardSSD_LRS"
    # With the default 30GB, docker will fail to load and export the image
    disk_size_gb = "64"
  }

  source_image_reference {
    publisher = "SUSE"
    offer     = "opensuse-leap-15-5"
    sku       = "gen2"
    version   = "latest"
  }
}

resource "null_resource" "example" {
  provisioner "remote-exec" {
    connection {
      host        = azurerm_linux_virtual_machine.gh-actions-build-monai-models-vm.public_ip_address
      user        = "adminuser"
      private_key = file("/tmp/ssh_id_gh")
    }

    inline = ["echo 'connected!'"]
  }
}

output "instance_public_ip" {
  description = "Public IP address"
  value       = azurerm_linux_virtual_machine.gh-actions-build-monai-models-vm.public_ip_address
}

